"""
Figure 6: CP Rank Selection Analysis

This figure demonstrates how R²X changes with increasing CP rank
while keeping the RISE rank fixed at 30.
"""

import pandas as pd
from matplotlib import pyplot as plt

from ..import_data import (
    add_cond_idxs,
    import_balf_covid,
    import_ligand_receptor_pairs,
)
from ..tensor import run_ccc_rise_workflow
from .common import load_or_compute_dataframe, subplotLabel


def compute_cp_rank_sweep(
    cp_ranks: list[int] | None = None,
    rise_rank: int = 30,
) -> pd.DataFrame:
    """Compute R2X across CP ranks for a fixed RISE rank."""
    if cp_ranks is None:
        cp_ranks = list(range(10, 101, 10))

    adata = import_balf_covid()
    adata_filtered = add_cond_idxs(adata, "sample")
    lr_pairs = import_ligand_receptor_pairs()

    r2x_results = []
    for cp_rank in cp_ranks:
        try:
            _, r2x = run_ccc_rise_workflow(
                adata_filtered,
                rise_rank=rise_rank,
                lr_pairs=lr_pairs,
                cp_rank=cp_rank,
                n_iter_max=100,
                tol=1e-6,
                random_state=42,
                doEmbedding=False,
            )
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
    df = df.dropna().sort_values("cp_rank")

    # Plot R²X curve
    ax.plot(
        df["cp_rank"],
        df["r2x"],
        "o-",
        linewidth=2,
        markersize=6,
        color="darkorange",
    )
    ax.set_xlabel("CP Rank", fontsize=14)
    ax.set_ylabel("R²X", fontsize=14)
    ax.set_title(f"CP Rank Selection (RISE Rank = {rise_rank})", fontsize=16)
    ax.grid(True, alpha=0.3)

    return fig
