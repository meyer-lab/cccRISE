"""
Figure 6: CP Rank Selection Analysis

This figure demonstrates how R²X changes with increasing CP rank
while keeping the RISE rank fixed at 30.
"""

import pandas as pd
from matplotlib import pyplot as plt
from tensorly.decomposition import parafac

from ..ccc import calculate_r2x
from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from ..tensor import calculate_interaction_tensor
from .common import load_or_compute_dataframe, subplotLabel


def compute_cp_rank_sweep(
    cp_ranks: list[int] | None = None,
    rise_rank: int = 30,
    n_iter_max: int = 1000,
    tol: float = 1e-6,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute R2X across CP ranks for a fixed RISE rank."""
    if cp_ranks is None:
        cp_ranks = list(range(10, 101, 10))

    adata = import_balf_covid()
    adata_filtered = add_cond_idxs(adata, "sample")
    lr_pairs = import_ligand_receptor_pairs()

    # Calculate interaction tensor ONCE for the fixed RISE rank
    interaction_tensor = calculate_interaction_tensor(
        adata_filtered, lr_pairs, rise_rank=rise_rank
    )

    r2x_results = []
    for cp_rank in cp_ranks:
        try:
            cp_weights, cp_factors = parafac(
                interaction_tensor,
                rank=cp_rank,
                n_iter_max=n_iter_max,
                tol=tol,
                normalize_factors=True,
                random_state=random_state,
            )
            r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensor)
            r2x_results.append({"cp_rank": cp_rank, "r2x": r2x})
        except Exception:
            r2x_results.append({"cp_rank": cp_rank, "r2x": float("nan")})

    return pd.DataFrame(r2x_results)


def makeFigure():
    """Generate Figure 6 showing CP rank selection analysis with fixed RISE rank."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    subplotLabel([ax])

    rise_rank = 30
    df = load_or_compute_dataframe(
        "fig6_cp_ranks.csv",
        compute_cp_rank_sweep,
        rise_rank=rise_rank,
    )

    ax.plot(
        df["cp_rank"],
        df["r2x"],
        marker="o",
        color="blue",
        label=f"RISE Rank {rise_rank}",
    )
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("R²X")
    ax.set_title("CP Rank Selection")
    ax.legend()

    return fig
