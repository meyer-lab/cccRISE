"""
Figure for CPD entry-masked imputation cross-validation rank selection.

Plot-only: the sweep (RISE fit, interaction-tensor construction, masked CP
cross-validation) lives in `cellcommunicationpf2.runners.run_cpd_rank`, which
writes ``cpd_rank_imputation_<dataset>_rise<rise_rank>.csv`` plus a YAML
sidecar into ``cellcommunicationpf2/data``. This module reads that CSV; if it
is missing, it invokes the runner to produce it.
"""

import matplotlib.pyplot as plt
import pandas as pd

from ..runners.datasets import get_dataset
from ..runners.run_cpd_rank import DEFAULT_RANKS, run_cpd_rank_selection
from .common import getSetup, load_or_compute_dataframe, subplotLabel

# Change this variable to switch dataset ("alad", "ibd").
SELECTED_DATASET = "alad"

# RISE rank the interaction tensor is built at, per dataset. CP rank selection
# is conditional on the already-selected RISE rank, so this must match the rank
# recorded in that dataset's RISE rank-selection sidecar.
DATASET_RISE_RANKS: dict[str, int] = {
    "alad": 25,
    "ibd": 25,
}
DEFAULT_N_MASKS = 5


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

    spec = get_dataset(SELECTED_DATASET)
    rise_rank = DATASET_RISE_RANKS[SELECTED_DATASET]

    df = load_or_compute_dataframe(
        spec.cpd_rank_filename(rise_rank),
        lambda: run_cpd_rank_selection(
            spec,
            rise_rank=rise_rank,
            ranks=DEFAULT_RANKS,
            n_masks=DEFAULT_N_MASKS,
        )[0],
    )

    plot_cp_rank_selection(
        df, ax[0], title=f"CPD Imputation $R^2X$ ({SELECTED_DATASET.upper()})"
    )
    return f
