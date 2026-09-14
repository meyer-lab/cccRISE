"""
Temporary supplied gene exclusion filter for vsparse.load_and_normalize. Eventually the API
will support a `gene_mask=` parameter, but this is a temporary workaround for now.
"""
from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from vsparse import _ivcsc
from vsparse._ops import accumulator_threads
from vsparse._rapid_load import (
    _FIELD_KEYS,
    _build_data,
    _build_selected_rows,
    _cell_totals,
    _filter_and_compact,
    _gene_detection_counts,
    _normalize_and_transform,
    _weighted_bincount,
)


def load_and_normalize_excluding(
    path: str | PathLike[str],
    *,
    min_cell_counts: float = 10.0,
    gene_threshold: float = 0.0,
    min_cells: int | None = None,
    obs_filter: Callable[[pd.DataFrame], object] | None = None,
    exclude_genes: Callable[[pd.DataFrame], np.ndarray] | None = None,
    x_key: str = "X",
) -> ad.AnnData:
    import h5py
    import hdf5plugin  # noqa: F401
    if min_cells is not None:
        if isinstance(min_cells, bool) or not isinstance(min_cells, int):
            raise TypeError("min_cells must be an integer or None")
        if min_cells < 0:
            raise ValueError("min_cells must be non-negative")

    with h5py.File(Path(path), "r") as f:
        g = f[x_key]
        shape = (int(g.attrs["shape"][0]), int(g.attrs["shape"][1]))
        indices_dtype = np.dtype(g.attrs["indices_dtype"])
        major_ptr = g["major_ptr"][...]
        values = g["values"][...]
        value_ptr = g["value_ptr"][...]
        packed = g["packed_indices"][...]

        kwargs = {k: ad.io.read_elem(f[k]) for k in _FIELD_KEYS if k in f}
        if "raw" in f:
            kwargs["raw"] = ad.io.read_elem(f["raw"])

    n_cells, n_genes = shape

    # The exclusion is resolved from `var` before any decoding, so a bad mask
    # fails before the expensive part rather than after it.
    excluded = _resolve_exclusion(exclude_genes, kwargs.get("var"), n_genes)

    cell_totals = _cell_totals(major_ptr, values, value_ptr)

    if obs_filter is None:
        cell_mask = cell_totals > min_cell_counts

        indices = _ivcsc.unpack_indices(value_ptr, packed, indices_dtype)
        del packed
        data = _build_data(values, value_ptr, indices.shape[0])
        row_indptr = value_ptr[major_ptr]
        n_selected = n_cells
    else:
        obs = kwargs.get("obs")
        if not isinstance(obs, pd.DataFrame):
            raise ValueError("obs_filter requires an obs table in the h5ad file")
        if not callable(obs_filter):
            raise TypeError("obs_filter must be callable or None")

        obs_mask = np.asarray(obs_filter(obs))
        if obs_mask.ndim != 1 or obs_mask.shape[0] != n_cells:
            raise ValueError(
                f"obs_filter must return a one-dimensional mask of length {n_cells}"
            )
        if obs_mask.dtype != np.bool_:
            raise ValueError("obs_filter must return a boolean mask")
        if not np.any(obs_mask):
            raise ValueError("obs_filter selected no cells")
        obs_mask = np.ascontiguousarray(obs_mask)

        selected_rows = np.nonzero(obs_mask)[0]
        cell_mask = cell_totals[obs_mask] > min_cell_counts
        row_indptr, indices, data = _build_selected_rows(
            major_ptr, values, value_ptr, packed, indices_dtype, obs_mask
        )
        del packed
        n_selected = selected_rows.shape[0]

    gene_totals_raw = _weighted_bincount(
        indices, data, n_genes, accumulator_threads(n_genes)
    )
    gene_mask = gene_totals_raw > (gene_threshold * n_selected)
    if min_cells is not None:
        gene_detection_counts = _gene_detection_counts(
            indices, data, n_genes, accumulator_threads(n_genes)
        )
        gene_mask &= gene_detection_counts >= min_cells

    # Drop the excluded genes here, so the
    # normalization below never sees their counts.
    gene_mask &= ~excluded
    if not gene_mask.any():
        raise ValueError("exclude_genes removed every gene")

    if obs_filter is None:
        metadata_cell_mask = cell_mask
    else:
        metadata_cell_mask = np.zeros(n_cells, dtype=np.bool_)
        metadata_cell_mask[selected_rows[cell_mask]] = True

    new_indptr, out_indices, out_data, kept_rows, n_kept_genes = _filter_and_compact(
        row_indptr, indices, data, cell_mask, gene_mask
    )
    del indices, data, row_indptr

    normalized = _normalize_and_transform(
        new_indptr, out_indices, out_data, n_kept_genes
    )
    X = csr_array(
        (normalized, out_indices, new_indptr), shape=(kept_rows.shape[0], n_kept_genes)
    )

    if "obs" in kwargs and "var" in kwargs:
        adata = ad.AnnData(**kwargs)  # ty: ignore[invalid-argument-type]
    else:
        adata = ad.AnnData(shape=shape, **kwargs)  # ty: ignore[invalid-argument-type]
    adata = adata[metadata_cell_mask, gene_mask].copy()
    adata.X = X

    return adata


def _resolve_exclusion(
    exclude_genes: Callable[[pd.DataFrame], np.ndarray] | None,
    var: object,
    n_genes: int,
) -> np.ndarray:
    """Turn the `exclude_genes` callable into a validated boolean mask."""
    if exclude_genes is None:
        return np.zeros(n_genes, dtype=np.bool_)
    if not isinstance(var, pd.DataFrame):
        raise TypeError("exclude_genes requires a var table in the h5ad file")
    if not callable(exclude_genes):
        raise TypeError("exclude_genes must be callable or None")

    mask = np.asarray(exclude_genes(var))
    if mask.ndim != 1 or mask.shape[0] != n_genes:
        raise ValueError(
            f"exclude_genes must return a one-dimensional mask of length {n_genes}"
        )
    if mask.dtype != np.bool_:
        raise ValueError("exclude_genes must return a boolean mask")
    return np.ascontiguousarray(mask)
