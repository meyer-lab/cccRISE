"""
Figure for CPD entry-masked imputation cross-validation rank selection.

Plots the sweep produced by ``uv run cccrise-cpdrank``; run that first.
"""

import matplotlib.pyplot as plt
import pandas as pd

from .common import getSetup, load_runner_csv, subplotLabel

# Change this to switch dataset; must match the runner's --dataset. The RISE
# rank the sweep was conditioned on is recorded in the sweep's sidecar.
SELECTED_DATASET = "alad"
SWEEP_CSV = f"cpd_rank_{SELECTED_DATASET}.csv"


def plot_cp_rank_selection(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str | None = None,
    plot_in_sample: bool = False,
) -> None:
    """Plot mean +/- SEM of held-out entry R2X across CP ranks from imputation CV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Rank' and 'HeldOutR2X' (and optionally
        'InSampleR2X') columns.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    title : str | None
        Optional title for the subplot.
    plot_in_sample : bool, default=False
        Whether to also plot in-sample R2X on the same axis.
    """
    agg_dict: dict[str, tuple[str, str]] = {
        "held_out_mean": ("HeldOutR2X", "mean"),
        "held_out_sem": ("HeldOutR2X", "sem"),
    }
    if "InSampleR2X" in df.columns:
        agg_dict["in_sample_mean"] = ("InSampleR2X", "mean")

    stats = df.groupby("Rank").agg(**agg_dict)

    ax.errorbar(
        stats.index,
        stats["held_out_mean"],
        yerr=stats["held_out_sem"],
        marker="o",
        capsize=3,
        label="Held-out $R^2X$",
    )
    if plot_in_sample and "in_sample_mean" in stats.columns:
        ax.plot(
            stats.index,
            stats["in_sample_mean"],
            marker="s",
            linestyle="--",
            alpha=0.7,
            label="In-sample $R^2X$",
        )
        ax.legend(frameon=True)

    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Held-out entry $R^2X$")
    if title:
        ax.set_title(title)


def makeFigure():
    ax, f = getSetup((4, 3.5), (1, 1))
    subplotLabel(ax)

    plot_cp_rank_selection(
        load_runner_csv(SWEEP_CSV),
        ax[0],
        title=f"CPD Imputation $R^2X$ ({SELECTED_DATASET.upper()})",
    )
    return f
