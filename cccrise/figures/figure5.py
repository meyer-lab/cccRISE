"""
Figure 5: RISE Rank Selection Analysis

This figure demonstrates how R²X changes with increasing PARAFAC2 (RISE) rank
to identify the optimal rank where R²X begins to flatline.
"""

import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd

from ..import_data import add_cond_idxs, import_balf_covid
from .common import getSetup, load_or_compute_dataframe


def compute_rise_rank_sweep(ranks: list[int] | None = None) -> pd.DataFrame:
    """Compute R2X across RISE ranks."""
    if ranks is None:
        ranks = list(range(1, 41, 5))

    adata = import_balf_covid()
    adata_filtered = add_cond_idxs(adata, "sample")

    r2x_results = []
    for rank in ranks:
        try:
            _, r2x = parafac2_nd(
                adata_filtered,
                rank=rank,
                n_iter_max=100,
                tol=1e-6,
                random_state=42,
            )
            r2x_results.append({"rank": rank, "r2x": r2x})
        except Exception:
            r2x_results.append({"rank": rank, "r2x": np.nan})

    return pd.DataFrame(r2x_results)


def makeFigure():
    """Generate Figure 5 showing RISE rank selection analysis."""
    ax, f = getSetup((6, 3), (1, 1))

    df = load_or_compute_dataframe("fig5_rise_ranks.csv", compute_rise_rank_sweep)
    df = df.dropna().sort_values("rank")

    # Plot R²X curve
    ax[0].plot(
        df["rank"],
        df["r2x"],
        "o-",
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    ax[0].set_xlabel("RISE Rank", fontsize=14)
    ax[0].set_ylabel("R²X", fontsize=14)
    ax[0].set_title("PARAFAC2 (RISE) Rank Selection Analysis", fontsize=16)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_ylim(0, 0.2)

    return f
