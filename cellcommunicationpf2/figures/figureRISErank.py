"""
Figure for RISE BiCV (bi-cross-validation) rank selection.

Plots the sweep produced by ``uv run ccc-rise-riserank``; run that first.
"""

import matplotlib.pyplot as plt
import pandas as pd

from .common import getSetup, load_runner_csv, subplotLabel

# Change these to switch dataset; must match the runner's --dataset.
SELECTED_DATASET = "alad"
SWEEP_CSV = f"rise_rank_{SELECTED_DATASET}.csv"


def plot_bicv_rank_selection(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str | None = None,
) -> None:
    """Plot mean +/- SEM of held-out block R2X across ranks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Rank' and 'HeldOutBlockR2X' columns.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    title : str | None
        Optional title for the subplot.
    """
    stats = df.groupby("Rank")["HeldOutBlockR2X"].agg(["mean", "sem"])
    ax.errorbar(
        stats.index,
        stats["mean"],
        yerr=stats["sem"],
        marker="o",
        capsize=3,
    )
    ax.set_xlabel("RISE Rank")
    ax.set_ylabel("Held-out block $R^2X$")
    if title:
        ax.set_title(title)


def makeFigure():
    ax, f = getSetup((4, 3.5), (1, 1))
    subplotLabel(ax)

    plot_bicv_rank_selection(
        load_runner_csv(SWEEP_CSV),
        ax[0],
        title=f"BiCV Held-out $R^2X$ ({SELECTED_DATASET.upper()})",
    )
    return f
