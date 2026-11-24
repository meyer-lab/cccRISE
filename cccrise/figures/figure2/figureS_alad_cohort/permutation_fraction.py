"""Distribution over LR pairs of the fraction of each LR pair's condition x
eigen-state entries exceeding their cell-permutation null at p < 0.05, ALAD
cohort."""

import numpy as np
import seaborn as sns

from ... import common, style
from ..permutation_lr_significance import lr_significance

AXES = ("square",)
KEY_HEADROOM = 1.25  # y-limit over the histogram's, for the legend


def plot(axs):
    """Fraction of significant condition x eigenstate entries per LR pair,
    against the 5% expected under the null."""
    ax = axs[0]
    fractions = lr_significance()["FractionSignificant_p05"].dropna()
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


def makeFigure():
    return common.standalone_fixed(__name__)
