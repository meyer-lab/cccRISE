"""Cell-permutation significance of each LR pair's communication scores: the
distribution over LR pairs of the fraction of condition x eigen-state entries
with p < 0.05, and each LR pair's mean observed score against its mean
permutation-null score."""

import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common, style

DATASET = "alad"
AXES = ("square", "square")
COLORBAR_TICKS = (0, common.SIG_ALPHA, 0.1, 0.2)
KEY_HEADROOM = 1.25  # y-limit over the histogram's, for the legend


def plot(axs):
    spec = DATASETS[DATASET]
    lr_table = common.load_runner_output(
        spec.significance_lr_filename(), VALIDATION_DIR
    )
    common.source_data(
        "permutation_significance_by_lr_pair",
        lr_table,
        "One row per LR pair, over every condition x sender eigen-state x "
        "receiver eigen-state entry: mean observed communication score, mean of "
        "its cell-permutation null, mean and median z-score against the null, "
        "fraction of entries with permutation p < 0.05 and < 0.01 (uncorrected), "
        "and the largest observed score. Each entry's observed score, p-value, "
        "and null mean and SD are deposited separately "
        "(significance_entries_alad.nc).",
    )
    _calibration(lr_table, axs[0])
    _observed_vs_null(lr_table, axs[1])


def _calibration(lr_table, ax):
    """Fraction of significant condition x eigenstate entries per LR pair,
    against the 5% expected under the null."""
    fractions = lr_table["FractionSignificant_p05"].dropna()
    x_max = max(0.25, np.ceil(fractions.max() * 20) / 20)
    sns.histplot(
        fractions,
        kde=True,
        ax=ax,
        color=style.COLORS["emphasis"],
        bins=np.linspace(0, x_max, 26),
        stat="density",
        alpha=0.4,
        edgecolor="white",
        linewidth=style.LINE_WIDTH_THIN,
    )
    null_line = common.reference_line(
        ax, common.SIG_ALPHA, horizontal=False, label="Permutation null (5%)"
    )
    null_line.set_ydata([0, 1 / KEY_HEADROOM])  # stops below the legend
    ax.set_xlim(0, x_max)
    ax.set_ylim(top=ax.get_ylim()[1] * KEY_HEADROOM)
    ax.set_xlabel("Fraction of entries significant\nper LR pair (p < 0.05)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")


def _observed_vs_null(lr_table, ax):
    """Mean observed score against the permutation-null mean per LR pair,
    colored by its fraction of significant entries, centered on the 5%
    expected under the null. Not the mean z-score: most entries' null SD is
    below `summarize_significance_lr`'s 1e-3 floor."""
    observed, null, fraction = (
        lr_table[c] for c in ("MeanObserved", "MeanNull", "FractionSignificant_p05")
    )
    points = ax.scatter(
        observed,
        null,
        c=fraction,
        cmap=style.CMAPS["diverging"],
        norm=TwoSlopeNorm(common.SIG_ALPHA, vmin=0, vmax=fraction.max()),
        s=style.POINT_SIZE_BACKGROUND,
        edgecolors="none",
        rasterized=True,
    )
    common.colorbar(ax, points, "Fraction significant").set_ticks(COLORBAR_TICKS)
    low = min(0.0, observed.min(), null.min())
    high = max(observed.max(), null.max()) * 1.05
    ax.plot(
        [low, high],
        [low, high],
        color=style.COLORS["reference"],
        linestyle=style.LINE_STYLE_REFERENCE,
        linewidth=style.LINE_WIDTH_THIN,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Mean observed\nLR pair score")
    ax.set_ylabel("Mean cell-permuted\nnull score")


def makeFigure():
    return common.standalone_fixed(__name__)
