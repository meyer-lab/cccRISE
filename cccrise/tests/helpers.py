"""Shared test fixtures."""

from pathlib import Path

import anndata
import numpy as np

from cccrise.ccc_rise import fit_rise
from cccrise.factors import save_ccc_rise_fit
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
    fit_adata, _ = fit_rise(
        add_cond_idxs(adata.copy(), condition_column),
        rank=rise_rank,
        n_iter_max=100,
        tolerance=1e-6,
        random_state=random_state,
        backend="cpu",
    )
    n_cond = fit_adata.obs[condition_column].nunique()
    placeholder_cp = {
        "cp_factors": [
            np.ones((n_cond, 1)),
            np.ones((rise_rank, 1)),
            np.ones((rise_rank, 1)),
            np.ones((1, 1)),
        ],
        "cp_weights": np.ones(1),
        "filtered_lr_pairs": ["placeholder"],
        "r2x_cp": 0.0,
    }
    save_ccc_rise_fit(path, fit_adata, placeholder_cp, condition_column, [])
    return fit_adata
