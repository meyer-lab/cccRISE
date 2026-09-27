"""Shared test fixtures."""

from pathlib import Path

import anndata
import numpy as np

from cccrise.ccc_rise import fit_rise
from cccrise.factors import (
    add_cell_projections,
    factors_to_xarray,
    save_factors_netcdf,
)
from cccrise.import_data import add_cond_idxs


def write_rise_reference(
    adata: anndata.AnnData,
    condition_column: str,
    rise_rank: int,
    path: Path,
    random_state: int = 0,
) -> anndata.AnnData:
    """A factors.nc holding a real RISE fit and rank-1 CP placeholders (a RISE
    reference never reads its CP portion); returns the fitted AnnData."""
    fit_adata, r2x_pf2 = fit_rise(
        add_cond_idxs(adata.copy(), condition_column),
        rank=rise_rank,
        n_iter_max=100,
        tolerance=1e-6,
        random_state=random_state,
        backend="cpu",
    )
    n_cond = fit_adata.obs[condition_column].nunique()
    ds = factors_to_xarray(
        cpd_factors=[
            np.ones((n_cond, 1)),
            np.ones((rise_rank, 1)),
            np.ones((rise_rank, 1)),
            np.ones((1, 1)),
        ],
        cpd_weights=np.ones(1),
        filtered_lr_pairs=["placeholder"],
        condition_names=np.unique(fit_adata.obs[condition_column]),
        gene_names=list(fit_adata.var_names),
        pf2_factors=[
            np.asarray(fit_adata.uns["Pf2_A"]),
            np.asarray(fit_adata.uns["Pf2_B"]),
            np.asarray(fit_adata.varm["Pf2_C"]),
        ],
        pf2_weights=fit_adata.uns["Pf2_weights"],
        r2x_pf2=r2x_pf2,
        attrs={"condition_key": condition_column, "complex_sep": "&"},
    )
    save_factors_netcdf(add_cell_projections(ds, fit_adata, obs_columns=[]), path)
    return fit_adata
