"""
Module for saving, loading, and projecting cell-cell communication factors
using xarray NetCDF objects.
"""

import hashlib
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from parafac2.utils import polar_factor

from .ccc_rise import add_cell_embeddings
from .import_data import add_cond_idxs
from .runners.datasets import DATASETS

DATA_DIR = Path(__file__).parent / "data"
COVID19_FACTORS_PATH = DATA_DIR / "bal_covid19_factors.nc"
ALAD_FACTORS_PATH = DATASETS["alad"].factors_path()
#: Factors-file attribute fingerprinting the fit's cells (see `save_ccc_rise_fit`).
CELL_IDS_ATTR = "cell_ids_sha256"
#: Every factors file loaded in this process, so a figure build can record
#: the fits its panels plot
LOADED_PATHS: set[Path] = set()


def _resolve_lr_symbols(
    filtered_lr_pairs: pd.DataFrame | list[str] | np.ndarray,
) -> np.ndarray:
    """Derive ligand-receptor pair label strings from any of the supported input forms."""
    if not isinstance(filtered_lr_pairs, pd.DataFrame):
        return np.asarray(filtered_lr_pairs)

    if "interaction_symbol" in filtered_lr_pairs.columns:
        return filtered_lr_pairs["interaction_symbol"].values
    if (
        "ligand" in filtered_lr_pairs.columns
        and "receptor" in filtered_lr_pairs.columns
    ):
        return (
            filtered_lr_pairs["ligand"].astype(str)
            + "_"
            + filtered_lr_pairs["receptor"].astype(str)
        ).values
    return np.array([f"pair_{i}" for i in range(len(filtered_lr_pairs))])


def _add_pf2_factors(
    data_vars: dict[str, tuple[list[str], np.ndarray]],
    coords: dict[str, Any],
    pf2_factors: list[np.ndarray] | None,
    pf2_weights: np.ndarray | None,
    gene_names: list[str] | np.ndarray | None,
) -> None:
    """Add PARAFAC2 factor matrices/weights to data_vars and coords, in place."""
    if pf2_factors is None or len(pf2_factors) < 3:
        return

    data_vars["pf2_A"] = (
        ["condition", "rise_component"],
        np.asarray(pf2_factors[0], dtype=np.float32),
    )
    data_vars["pf2_B"] = (
        ["rise_component", "rise_component_dim2"],
        np.asarray(pf2_factors[1], dtype=np.float32),
    )
    coords["rise_component_dim2"] = np.arange(1, pf2_factors[1].shape[1] + 1)
    if gene_names is not None:
        data_vars["pf2_C"] = (
            ["gene", "rise_component"],
            np.asarray(pf2_factors[2], dtype=np.float32),
        )
    if pf2_weights is not None:
        data_vars["pf2_weights"] = (
            ["rise_component"],
            np.asarray(pf2_weights, dtype=np.float32),
        )


def factors_to_xarray(
    cpd_factors: list[np.ndarray],
    cpd_weights: np.ndarray,
    filtered_lr_pairs: pd.DataFrame | list[str] | np.ndarray,
    condition_names: list[str] | np.ndarray,
    gene_names: list[str] | np.ndarray | None = None,
    pf2_factors: list[np.ndarray] | None = None,
    pf2_weights: np.ndarray | None = None,
    r2x_cp: float | None = None,
    r2x_pf2: float | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """
    Convert PARAFAC2 and CP decomposition factor matrices to a structured xarray Dataset.

    Parameters
    ----------
    cpd_factors : list[np.ndarray]
        List of 4 CP factor matrices: [A (condition), B (sender), C (receiver), D (LR)].
    cpd_weights : np.ndarray
        Component scaling weights for CP decomposition.
    filtered_lr_pairs : pd.DataFrame or list[str] or np.ndarray
        Ligand-receptor pair symbols matching dimension 3 of the communication tensor.
    condition_names : list[str] or np.ndarray
        Labels for conditions / samples matching dimension 0.
    gene_names : list[str] or np.ndarray, optional
        List of gene identifiers.
    pf2_factors : list[np.ndarray], optional
        List of PARAFAC2 factor matrices [A_pf2, B_pf2, C_pf2].
    pf2_weights : np.ndarray, optional
        PARAFAC2 component weights.
    r2x_cp : float, optional
        Variance explained by CP decomposition.
    r2x_pf2 : float, optional
        Variance explained by PARAFAC2 decomposition.
    attrs : dict, optional
        Additional metadata dictionary for reproducibility.

    Returns
    -------
    xr.Dataset
        A fully-annotated xarray Dataset holding the factor matrices and coordinates.
    """
    lr_symbols = _resolve_lr_symbols(filtered_lr_pairs)

    condition_names = np.asarray(condition_names)
    rise_rank = cpd_factors[1].shape[0]
    cp_rank = cpd_factors[0].shape[1]

    data_vars: dict[str, tuple[list[str], np.ndarray]] = {
        "cp_A": (
            ["condition", "cp_component"],
            np.asarray(cpd_factors[0], dtype=np.float32),
        ),
        "cp_B": (
            ["rise_component", "cp_component"],
            np.asarray(cpd_factors[1], dtype=np.float32),
        ),
        "cp_C": (
            ["rise_component", "cp_component"],
            np.asarray(cpd_factors[2], dtype=np.float32),
        ),
        "cp_D": (
            ["lr_pair", "cp_component"],
            np.asarray(cpd_factors[3], dtype=np.float32),
        ),
        "cp_weights": (["cp_component"], np.asarray(cpd_weights, dtype=np.float32)),
    }

    coords: dict[str, Any] = {
        "condition": condition_names,
        "rise_component": np.arange(1, rise_rank + 1),
        "cp_component": np.arange(1, cp_rank + 1),
        "lr_pair": lr_symbols,
    }

    if gene_names is not None:
        coords["gene"] = np.asarray(gene_names)

    _add_pf2_factors(data_vars, coords, pf2_factors, pf2_weights, gene_names)

    ds_attrs = {
        "rise_rank": rise_rank,
        "cp_rank": cp_rank,
    }
    if r2x_cp is not None:
        ds_attrs["r2x_cp"] = float(r2x_cp)
    if r2x_pf2 is not None:
        ds_attrs["r2x_pf2"] = float(r2x_pf2)
    if attrs is not None:
        ds_attrs.update(attrs)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_attrs)


def save_factors_netcdf(ds: xr.Dataset, filepath: str | Path) -> None:
    """Save an xarray Dataset of factors to a NetCDF file, ensuring float
    arrays are float32, except for float16 ones (the PaCMAP coordinates
    `save_ccc_rise_fit` stores).

    float16 isn't part of the classic netCDF-4 data model, so if any
    variable was quantized to float16 the file is written via the
    `h5netcdf` engine (HDF5-native, supports float16) instead of the
    default `netcdf4` engine.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds_to_save = ds.copy()
    has_float16 = False
    for var_name in ds_to_save.data_vars:
        dtype = ds_to_save[var_name].dtype
        if dtype == np.float16:
            has_float16 = True
        elif np.issubdtype(dtype, np.floating):
            ds_to_save[var_name] = ds_to_save[var_name].astype(np.float32)

    if has_float16:
        ds_to_save.to_netcdf(path, engine="h5netcdf", invalid_netcdf=True)
    else:
        ds_to_save.to_netcdf(path)


def load_factors_netcdf(filepath: str | Path) -> xr.Dataset:
    """Load an xarray Dataset of factors from a NetCDF file.

    Always uses the `h5netcdf` engine: it reads both plain netCDF-4 files
    and ones written with float16 arrays (see
    `save_factors_netcdf`). The default `netcdf4` engine can't represent
    float16 and, worse, doesn't fail cleanly when asked to open such a
    file, so it's avoided entirely here rather than used as a fallback.

    Loads eagerly (`load_dataset`, not `open_dataset`) so the returned
    Dataset stays usable after the underlying file is closed -- needed to
    read a file and then overwrite it in the same process.
    """
    LOADED_PATHS.add(Path(filepath).resolve())
    return xr.load_dataset(Path(filepath), engine="h5netcdf")


def save_ccc_rise_fit(
    path: str | Path,
    adata: anndata.AnnData,
    cp_fit: dict[str, Any],
    condition_column: str,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write a CCC-RISE fit as a factors file -- the file
    `load_factors_to_anndata` reads.

    Per-cell projections are not stored: `compute_cell_projections`
    recomputes them from the RISE factors and the raw data, which the file
    fingerprints (`CELL_IDS_ATTR`, `gene_means`) so a later recompute can
    confirm it has the fit's data. PaCMAP coordinates, which cannot be
    recomputed reproducibly, are stored (float16) when `adata` has them.

    Parameters
    ----------
    path : str | Path
        NetCDF destination.
    adata : anndata.AnnData
        The RISE-fit AnnData (`ccc_rise.fit_rise` / `run_ccc_rise_workflow`).
    cp_fit : dict[str, Any]
        A `ccc_rise.fit_ccc_cp` / `run_ccc_rise_workflow` result on `adata`.
    condition_column : str
        Column in `adata.obs` defining the conditions.
    attrs : dict, optional
        Extra file attributes (e.g. dataset name, `complex_sep`).
    """
    ds = factors_to_xarray(
        cpd_factors=cp_fit["cp_factors"],
        cpd_weights=cp_fit["cp_weights"],
        filtered_lr_pairs=cp_fit["filtered_lr_pairs"],
        # The row order `add_cond_idxs` gave the fit's condition mode
        condition_names=np.unique(adata.obs[condition_column].unique()),
        gene_names=list(adata.var_names),
        pf2_factors=[
            adata.uns["Pf2_A"],
            adata.uns["Pf2_B"],
            np.asarray(adata.varm["Pf2_C"]),
        ],
        pf2_weights=adata.uns["Pf2_weights"],
        r2x_cp=cp_fit["r2x_cp"],
        r2x_pf2=adata.uns["r2x_pf2"],
        attrs={"condition_key": condition_column} | (attrs or {}),
    )
    ds = add_fit_fingerprint(ds, adata)
    if "PaCMAP" in adata.obsm:
        ds["cell_pacmap"] = (
            ["cell", "pacmap_dim"],
            np.asarray(adata.obsm["PaCMAP"], dtype=np.float16),
        )
    save_factors_netcdf(ds, path)


def add_fit_fingerprint(ds: xr.Dataset, adata: anndata.AnnData) -> xr.Dataset:
    """Record in a factors Dataset which data its RISE fit was made on: a
    hash of `adata`'s cell IDs (`CELL_IDS_ATTR`) and its per-gene centering
    (`gene_means`), which `compute_cell_projections` checks."""
    ds = ds.copy()
    ds.attrs[CELL_IDS_ATTR] = _cell_ids_digest(adata)
    ds["gene_means"] = (["gene"], _gene_means(adata))
    return ds


def _cell_ids_digest(adata: anndata.AnnData) -> str:
    """SHA-256 of `adata`'s cell IDs, in order."""
    return hashlib.sha256("\n".join(map(str, adata.obs_names)).encode()).hexdigest()


def _gene_means(adata: anndata.AnnData) -> np.ndarray:
    """The per-gene centering `condition_matrices` and `parafac2` apply."""
    if "means" in adata.var:
        return adata.var["means"].to_numpy()
    return np.zeros(adata.n_vars)


def _check_fit_inputs(factors_ds: xr.Dataset, adata: anndata.AnnData) -> None:
    """Raise unless `adata` has the cells, genes and gene centering the
    stored RISE fit was made on (`add_fit_fingerprint`)."""
    if not np.array_equal(
        factors_ds.coords["gene"].values.astype(str),
        np.asarray(adata.var_names).astype(str),
    ):
        raise ValueError("`adata`'s genes do not match the RISE fit's.")
    if factors_ds.attrs[CELL_IDS_ATTR] != _cell_ids_digest(adata):
        raise ValueError("`adata`'s cells do not match the RISE fit's (IDs or order).")
    if not np.allclose(
        factors_ds["gene_means"].values, _gene_means(adata), rtol=1e-5, atol=1e-7
    ):
        raise ValueError(
            "`adata`'s gene means differ from the RISE fit's; the raw data or "
            "its preprocessing changed since the fit."
        )


def compute_cell_projections(
    factors_ds: xr.Dataset, adata: anndata.AnnData
) -> np.ndarray:
    """Per-cell RISE projections of `adata` under the fit stored in
    `factors_ds`.

    For each condition k this is the projection step `parafac2` ends a fit
    with: the polar factor (`parafac2.utils.polar_factor`) of the centered
    X_k @ C diag(w * A_k) B^T, so it reproduces the fit's own projections. Works on sparse `adata.X` without
    densifying it (the centering is applied to the rank-sized product).

    Parameters
    ----------
    factors_ds : xr.Dataset
        A factors file holding `pf2_A`/`pf2_B`/`pf2_C`/`pf2_weights`.
    adata : anndata.AnnData
        The data the fit was made on, with `obs["condition_unique_idxs"]`
        (`add_cond_idxs`); checked against the file's fingerprints.

    Returns
    -------
    np.ndarray
        (n_cells, rise_rank) float32 projections, in `adata`'s cell order.
    """
    _check_fit_inputs(factors_ds, adata)
    A, B, C, weights = (
        factors_ds[f"pf2_{m}"].values.astype(np.float64)
        for m in ("A", "B", "C", "weights")
    )
    means = _gene_means(adata)
    cond_idx = adata.obs["condition_unique_idxs"].to_numpy().astype(int)
    assert cond_idx.max() + 1 == A.shape[0], "`adata` has a different condition count"

    projections = np.zeros((adata.n_obs, A.shape[1]), dtype=np.float32)
    for k in range(A.shape[0]):
        rows = cond_idx == k
        W = (C * (weights * A[k])) @ B.T
        M = np.asarray(cast(Any, adata.X)[rows] @ W) - means @ W
        projections[rows] = polar_factor(M)
    return projections


def _add_cp_factors(out: anndata.AnnData, factors_ds: xr.Dataset) -> anndata.AnnData:
    """CP factors, weights, LR-pair labels and R2X into `out.uns`."""
    for key in "ABCD":
        out.uns[key] = factors_ds[f"cp_{key}"].values
    out.uns["weights"] = factors_ds["cp_weights"].values
    out.uns["lr_pairs"] = factors_ds.coords["lr_pair"].values
    out.uns["r2x"] = float(factors_ds.attrs.get("r2x_cp", 0.0))
    return out


def load_factors_to_anndata(
    factors_ds: xr.Dataset | str | Path,
    adata: anndata.AnnData | None = None,
    condition_column: str | None = None,
    do_embedding: bool = False,
    random_state: int | None = 42,
) -> anndata.AnnData:
    """
    Populate or construct an AnnData object with decomposition factors from an xarray Dataset.

    If `adata` is provided (the single-cell data the fit was made on),
    per-cell RISE projections (`obsm["projections"]`, recomputed by
    `compute_cell_projections`), sender loadings (`obsm["sc_B"]`), receiver
    loadings (`obsm["rc_C"]`), and 2D PaCMAP coordinates (stored, or fit when
    `do_embedding`) are added.

    If `adata` is None, creates a lightweight AnnData holding factor matrices
    in `.uns` and condition annotations in `.obs`.

    Parameters
    ----------
    factors_ds : xr.Dataset or str or Path
        Factor dataset or path to NetCDF file.
    adata : anndata.AnnData, optional
        Single-cell AnnData to project onto.
    condition_column : str, optional
        Condition/sample column in `adata.obs`. If None, uses `factors_ds.attrs.get("condition_key")`.
    do_embedding : bool, default=False
        Whether to fit a PaCMAP 2D embedding when the file stores none.
    random_state : int, optional
        Random seed for PaCMAP.

    Returns
    -------
    anndata.AnnData
        AnnData object populated with factors and single-cell projections.
    """
    if isinstance(factors_ds, (str, Path)):
        factors_ds = load_factors_netcdf(factors_ds)

    cond_key = condition_column or factors_ds.attrs.get("condition_key", "sample")

    if adata is None:
        conditions = factors_ds.coords["condition"].values
        out = anndata.AnnData(
            obs=pd.DataFrame({cond_key: conditions}, index=conditions),
            shape=(len(conditions), 0),
        )
        return _add_cp_factors(out, factors_ds)

    out = add_cond_idxs(_add_cp_factors(adata.copy(), factors_ds), cond_key)

    # Subset `adata` to the fit's genes when it holds them all
    ds_genes = factors_ds.coords["gene"].values
    if not np.array_equal(out.var_names, ds_genes) and set(ds_genes) <= set(
        out.var_names
    ):
        out = out[:, ds_genes].copy()

    out.obsm["projections"] = compute_cell_projections(factors_ds, out)
    has_pacmap = "cell_pacmap" in factors_ds
    out = add_cell_embeddings(
        out,
        factors_ds["cp_B"].values,
        factors_ds["cp_C"].values,
        do_embedding and not has_pacmap,
        random_state,
    )
    if has_pacmap:
        out.obsm["PaCMAP"] = np.asarray(
            factors_ds["cell_pacmap"].values, dtype=np.float32
        )
    return out


def clean_memory() -> None:
    """Explicitly garbage collect and release CuPy device and pinned host memory pools."""
    import gc
    import importlib

    gc.collect()
    try:
        cp = importlib.import_module("cupy")

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # no CuPy, or CuPy installed without a visible GPU
        return


def load_or_compute_dataframe(
    filename: str | Path,
    compute_fn,
    *args,
    recompute: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a DataFrame from a cached file in DATA_DIR or an absolute path,
    or compute it via `compute_fn(*args, **kwargs)` and cache it to disk.

    Supports .csv and .parquet formats based on file extension.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = DATA_DIR / path

    if path.exists() and not recompute:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    df: pd.DataFrame = compute_fn(*args, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    clean_memory()
    return df
