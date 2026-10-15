"""Each CP component's top-LR-pair set p (`lr_permutation`) as the set grows
from its single most heavily weighted pair to `MAX_TOP`, with the main
figure's set size marked."""

import numpy as np

from ... import common, style
from .._common import COMPONENT
from ..coordinated_changes import N_TOP
from ..lr_permutation import MAX_TOP, permutation_analysis, set_pvalue

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    sizes = np.arange(1, MAX_TOP + 1)
    for k, component in permutation_analysis().items():
        main = k == COMPONENT
        logp = -np.log10([set_pvalue(component, int(n)) for n in sizes])
        ax.plot(
            sizes,
            logp,
            color=style.COLORS["emphasis" if main else "muted"],
            lw=style.LINE_WIDTH if main else style.LINE_WIDTH_THIN,
            zorder=3 if main else 2,
        )
        ax.text(
            sizes[-1] + 0.5,
            logp[-1],
            str(k),
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            color=style.COLORS["emphasis" if main else "ink"],
        )
    common.reference_line(ax, -np.log10(common.SIG_ALPHA))
    common.reference_line(ax, N_TOP, horizontal=False, linestyle=":")
    ax.set(xlabel="Top-N LR pairs (N)", ylabel="−log$_{10}$(top-N set p)")


def makeFigure():
    return common.standalone_fixed(__name__)
