"""
Module for saving, loading, and projecting cell-cell communication factors
using xarray NetCDF objects.
"""

from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from pacmap import PaCMAP

from .import_data import add_cond_idxs, condition_matrices

DATA_DIR = Path(__file__).parent / "data"
COVID19_FACTORS_PATH = DATA_DIR / "bal_covid19_factors.nc"
ALAD_FACTORS_PATH = DATA_DIR / "bal_alad_factors.nc"


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
    """Save an xarray Dataset of factors to a NetCDF file, ensuring float arrays are float32."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds_to_save = ds.copy()
    for var_name in ds_to_save.data_vars:
        if np.issubdtype(ds_to_save[var_name].dtype, np.floating):
            ds_to_save[var_name] = ds_to_save[var_name].astype(np.float32)
    ds_to_save.to_netcdf(path)


def load_factors_netcdf(filepath: str | Path) -> xr.Dataset:
    """Load an xarray Dataset of factors from a NetCDF file."""
    return xr.open_dataset(Path(filepath))


def load_factors_to_anndata(
    factors_ds: xr.Dataset | str | Path,
    adata: anndata.AnnData | None = None,
    condition_column: str | None = None,
    do_embedding: bool = False,
    random_state: int | None = 0,
) -> anndata.AnnData:
    """
    Populate or construct an AnnData object with decomposition factors from an xarray Dataset.

    If `adata` is provided (e.g. raw/filtered single-cell data), single-cell factor
    projections (`obsm["projections"]`), sender loadings (`obsm["sc_B"]`), receiver
    loadings (`obsm["rc_C"]`), and optional 2D PaCMAP coordinates are computed.

    If `adata` is None, creates a lightweight AnnData holding factor matrices in `.uns`
    and condition annotations in `.obs`.

    Parameters
    ----------
    factors_ds : xr.Dataset or str or Path
        Factor dataset or path to NetCDF file.
    adata : anndata.AnnData, optional
        Single-cell AnnData to project onto.
    condition_column : str, optional
        Condition/sample column in `adata.obs`. If None, uses `factors_ds.attrs.get("condition_key")`.
    do_embedding : bool, default=False
        Whether to compute PaCMAP 2D embedding on single-cell projections.
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

    A_cp = factors_ds["cp_A"].values
    B_cp = factors_ds["cp_B"].values
    C_cp = factors_ds["cp_C"].values
    D_cp = factors_ds["cp_D"].values
    weights_cp = factors_ds["cp_weights"].values
    lr_pairs = factors_ds.coords["lr_pair"].values
    r2x_cp = float(factors_ds.attrs.get("r2x_cp", 0.0))

    if adata is None:
        # Construct lightweight factor container AnnData
        obs_df = pd.DataFrame(
            {cond_key: factors_ds.coords["condition"].values},
            index=factors_ds.coords["condition"].values,
        )
        out = anndata.AnnData(
            obs=obs_df,
            shape=(len(factors_ds.coords["condition"]), 0),
        )
        out.uns["A"] = A_cp
        out.uns["B"] = B_cp
        out.uns["C"] = C_cp
        out.uns["D"] = D_cp
        out.uns["weights"] = weights_cp
        out.uns["lr_pairs"] = lr_pairs
        out.uns["r2x"] = r2x_cp
        return out

    out = adata.copy()
    out.uns["A"] = A_cp
    out.uns["B"] = B_cp
    out.uns["C"] = C_cp
    out.uns["D"] = D_cp
    out.uns["weights"] = weights_cp
    out.uns["lr_pairs"] = lr_pairs
    out.uns["r2x"] = r2x_cp

    # Ensure condition indexing
    out = add_cond_idxs(out, cond_key)

    # Compute cell projections if PARAFAC2 factors are present
    rise_rank = int(factors_ds.attrs.get("rise_rank", B_cp.shape[0]))
    if "pf2_A" in factors_ds and "pf2_B" in factors_ds and "pf2_C" in factors_ds:
        A_pf2 = factors_ds["pf2_A"].values
        B_pf2 = factors_ds["pf2_B"].values
        C_pf2 = factors_ds["pf2_C"].values

        # Project matrices per condition
        X_list = condition_matrices(out)
        lhs_all = np.einsum("ik,gk,rk->igr", A_pf2, C_pf2, B_pf2, optimize=True)

        sg_index = out.obs["condition_unique_idxs"].values
        out.obsm["projections"] = np.zeros((out.shape[0], rise_rank))

        for i, mat in enumerate(X_list):
            M = mat @ lhs_all[i]
            G = M.T @ M
            _, V = np.linalg.eigh(G)
            MV = M @ V
            col_norms = np.linalg.norm(MV, axis=0, keepdims=True)
            safe_norms = np.where(col_norms > 1e-10, col_norms, 1.0)
            proj = (MV / safe_norms) @ V.T
            out.obsm["projections"][sg_index == i, :] = proj

        out.obsm["sc_B"] = out.obsm["projections"] @ B_cp
        out.obsm["rc_C"] = out.obsm["projections"] @ C_cp

        if do_embedding:
            pcm = PaCMAP(random_state=random_state)
            out.obsm["PaCMAP"] = pcm.fit_transform(out.obsm["projections"])

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
    except (ImportError, AttributeError):
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
