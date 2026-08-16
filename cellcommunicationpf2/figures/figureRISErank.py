"""
Figure for RISE BiCV (bi-cross-validation) rank selection.
"""

from collections.abc import Callable

import anndata
import matplotlib.pyplot as plt
import pandas as pd

from ..import_data import import_alad, import_balf_covid
from ..rank_selection import run_bicv
from .common import getSetup, load_or_compute_dataframe, subplotLabel

# Easily selectable dataset registry: (loader_func, condition_column)
DATASET_LOADERS: dict[str, tuple[Callable[[], anndata.AnnData], str]] = {
    "covid": (import_balf_covid, "sample"),
    "alad": (import_alad, "dsco_id"),
}

# Change this variable to switch dataset
SELECTED_DATASET = "covid"
DEFAULT_RANKS = [30, 35, 40]


def compute_bicv_dataset(
    dataset_name: str = SELECTED_DATASET,
    ranks: list[int] | None = None,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Compute BiCV reconstruction metrics for a given dataset across ranks.

    Parameters
    ----------
    dataset_name : str, default=SELECTED_DATASET
        Name of the dataset in `DATASET_LOADERS`.
    ranks : list[int] | None
        List of ranks to evaluate. Defaults to DEFAULT_RANKS.
    n_folds : int, default=3
        Number of cross-validation folds.

    Returns
    -------
    pd.DataFrame
        DataFrame with BiCV reconstruction metrics across ranks and folds.
    """
    if ranks is None:
        ranks = DEFAULT_RANKS
    loader, cond_col = DATASET_LOADERS[dataset_name]
    adata = loader()
    return run_bicv(adata, ranks=ranks, condition_column=cond_col, n_folds=n_folds)


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

    df = load_or_compute_dataframe(
        f"rise_rank_bicv_{SELECTED_DATASET}.csv",
        lambda: compute_bicv_dataset(dataset_name=SELECTED_DATASET, n_folds=3),
    )

    plot_bicv_rank_selection(
        df, ax[0], title=f"BiCV Held-out $R^2X$ ({SELECTED_DATASET.upper()})"
    )
    return f
