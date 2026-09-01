"""
Figure for RISE BiCV (bi-cross-validation) rank selection.

Plot-only: the sweep itself lives in
`cellcommunicationpf2.runners.run_rise_rank`, which writes
``rise_rank_bicv_<dataset>.csv`` (plus a YAML sidecar documenting the
parameters) into ``cellcommunicationpf2/data``. This module reads that CSV;
if it is missing, it invokes the runner to produce it, so `make` still works
from a clean checkout.
"""

import matplotlib.pyplot as plt
import pandas as pd

from ..runners.datasets import get_dataset
from ..runners.run_rise_rank import DEFAULT_RANKS, run_rise_rank_selection
from .common import getSetup, load_or_compute_dataframe, subplotLabel

# Change this variable to switch dataset ("alad", "ibd").
SELECTED_DATASET = "alad"
DEFAULT_N_FOLDS = 5


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

    spec = get_dataset(SELECTED_DATASET)
    df = load_or_compute_dataframe(
        spec.rise_rank_filename(),
        lambda: run_rise_rank_selection(
            spec, ranks=DEFAULT_RANKS, n_folds=DEFAULT_N_FOLDS
        )[0],
    )

    plot_bicv_rank_selection(
        df, ax[0], title=f"BiCV Held-out $R^2X$ ({SELECTED_DATASET.upper()})"
    )
    return f
