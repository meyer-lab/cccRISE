"""Load a VCSC/VCSR-backed `.h5ad`, filter it, then normalize what is left.

This replaces `vsparse.load_and_normalize`, which fuses reading, QC filtering
and normalization into one call and so gives a caller no point at which to
intervene. Gene exclusion -- dropping the mitochondrial and ribosomal genes
(`cccrise.gene_filters`) -- has to happen *before* the depth divisor is
computed, because those genes carry a large share of each cell's counts and
including them changes every cell's scale factor. That is the whole reason a
separate loader exists.

The sequence here is the one `vsparse`'s declarative recipes make expressible:

    load_packed  ->  VCSC/VCSR-backed AnnData, values still value-compressed
    index        ->  drop QC-failing cells, low-count genes, excluded genes
    .normalized("parafac2")
                 ->  statistics computed against the *surviving* matrix
    to_scipy_sparse()
                 ->  the sparse, uncentered term, plus `means` to subtract

`RECIPES["parafac2"]` is `y = (g(x*a*b) - c) * s` with `a = median(depth)/depth`,
`b = 1/sum_i(x[i,j] * a[i])`, `g = log10(1 + 1000x)`, `c` the per-gene mean and
`s = 1` -- exactly what `prepare_dataset`/`load_and_normalize` compute. `c` is
kept out of `X` and returned as `var["means"]`, matching the convention
`condition_matrices` and `parafac2` already follow: a sparse matrix plus a
per-gene mean vector, centered only once a condition block is made dense.
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import vsparse

__all__ = ["load_normalized"]


def load_normalized(
    path: str | PathLike[str],
    *,
    min_cell_counts: float = 10.0,
    gene_threshold: float = 0.0,
    min_cells: int | None = None,
    exclude_genes: Callable[[pd.DataFrame], np.ndarray] | None = None,
    obs_filter: Callable[[pd.DataFrame], Any] | None = None,
    recipe: str | vsparse.Recipe = "parafac2",
    dtype: Any = np.float32,
    x_key: str = "X",
) -> anndata.AnnData:
    """Load, filter and normalize an IVCSC/IVCSR-backed `.h5ad`.

    Reproduces `vsparse.load_and_normalize` (and so
    `parafac2.normalize.prepare_dataset`) and adds `exclude_genes`.

    Filter thresholds are measured on raw counts, before anything is dropped,
    so they do not depend on the order the masks are applied:

    - cells with total counts <= `min_cell_counts` are dropped;
    - genes with total counts <= `gene_threshold * n_cells` are dropped, where
      `n_cells` counts the cells surviving `obs_filter` but *not* yet
      `min_cell_counts` -- matching `prepare_dataset`;
    - when `min_cells` is given, genes detected in fewer than that many cells
      are dropped as well;
    - genes selected by `exclude_genes` are dropped unconditionally.

    Normalization statistics are then computed against the surviving matrix,
    which is the point: an excluded gene contributes to no cell's depth.

    Parameters
    ----------
    exclude_genes
        Callable given `var` and returning a boolean mask of genes to drop,
        e.g. `cccrise.gene_filters.mito_ribo_mask`. Resolved before any
        reduction runs, so a bad mask fails immediately.
    obs_filter
        Callable given `obs` and returning a boolean cell mask, applied before
        every threshold above is measured.
    recipe
        A `vsparse` recipe name or `Recipe`. The default, `"parafac2"`, is the
        normalization the rest of this package assumes.
    dtype
        Dtype of the returned `X.data`. The default matches the float32
        `load_and_normalize` returned, which is also what the stored values
        already are -- materializing at float64 would double the largest array
        in the process (14.6 GB -> 29.2 GB on the IBDverse cohort) to carry
        digits the input never had. Pass `np.float64` when a downstream step
        needs the wider accumulator.

    Returns
    -------
    anndata.AnnData
        `X` a `csr_array`/`csc_array` of the uncentered normalized values, and
        `var["means"]` the per-gene mean to subtract when a block is densified.
    """
    packed = vsparse.load_packed(path, x_key=x_key)
    if packed.X is None:
        raise ValueError(f"{path} has no {x_key} to normalize")
    n_cells, n_genes = packed.X.shape

    excluded = _resolve_mask(
        exclude_genes, cast(pd.DataFrame, packed.var), n_genes, "exclude_genes"
    )

    if obs_filter is not None:
        obs_mask = _resolve_mask(
            obs_filter, cast(pd.DataFrame, packed.obs), n_cells, "obs_filter"
        )
        if not obs_mask.any():
            raise ValueError("obs_filter selected no cells")
        packed = packed[obs_mask]

    arr = cast(Any, packed.X)
    n_selected = arr.shape[0]

    # Both reductions read the value-compressed form directly; neither
    # materializes the ~O(nnz) data array.
    cell_mask = np.asarray(arr.sum(axis=1)).reshape(-1) > min_cell_counts
    gene_mask = np.asarray(arr.sum(axis=0)).reshape(-1) > gene_threshold * n_selected
    if min_cells is not None:
        if isinstance(min_cells, bool) or not isinstance(min_cells, int):
            raise TypeError("min_cells must be an integer or None")
        if min_cells < 0:
            raise ValueError("min_cells must be non-negative")
        gene_mask &= np.asarray(arr.getnnz(axis=0)).reshape(-1) >= min_cells
    gene_mask &= ~excluded

    if not cell_mask.any():
        raise ValueError("every cell was filtered out")
    if not gene_mask.any():
        raise ValueError("every gene was filtered out")

    kept = packed[cell_mask, gene_mask]
    del packed, arr

    # `recalculate=True` is the default, but say so: the statistics must come
    # from this subset, not from the pre-filter population. `__getitem__` marks
    # inherited statistics `stale` precisely because reusing them here would be
    # the silent-wrong-answer case.
    view = kept.normalized(recipe, recalculate=True)

    adata = anndata.AnnData(
        X=view.to_scipy_sparse(dtype=dtype),
        obs=cast(pd.DataFrame, kept.obs).copy(),
        var=cast(pd.DataFrame, kept.var).copy(),
        uns=dict(kept.uns),
    )
    # Assigned rather than passed to the constructor: `normalized()` records
    # the recipe's per-gene b/c/s into `varm`, and AnnData's constructor
    # signature does not admit that mapping's value types.
    for key, value in kept.obsm.items():
        adata.obsm[key] = value
    for key, value in kept.varm.items():
        adata.varm[key] = value
    adata.var["means"] = np.asarray(view.means, dtype=np.float64)
    return adata


def _resolve_mask(
    fn: Callable[[pd.DataFrame], Any] | None,
    table: pd.DataFrame,
    length: int,
    name: str,
) -> np.ndarray:
    """Validate a caller-supplied mask callable before any expensive work."""
    if fn is None:
        return np.zeros(length, dtype=np.bool_)
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{name} requires the matching obs/var table")
    if not callable(fn):
        raise TypeError(f"{name} must be callable or None")

    mask = np.asarray(fn(table))
    if mask.ndim != 1 or mask.shape[0] != length:
        raise ValueError(
            f"{name} must return a one-dimensional mask of length {length}"
        )
    if mask.dtype != np.bool_:
        raise ValueError(f"{name} must return a boolean mask")
    return np.ascontiguousarray(mask)
