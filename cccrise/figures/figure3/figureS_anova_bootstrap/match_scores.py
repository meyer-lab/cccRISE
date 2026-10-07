"""How well each replicate's CP component matched the full-cohort component on
every resampling axis -- the control for the effect-size panels: a wide
effect-size distribution with low match scores means the component was not
re-identified, not that its association was unstable."""

import numpy as np
import seaborn as sns

from ... import common, style
from . import _common as S

#: Match score below which a replicate's component is not the reference one
#: in any useful sense.
MATCH_SCORE_FLOOR = 0.7


def plot(fig):
    ax = fig.subplots()
    sns.boxplot(
        data=S.replicates(),
        x="Component",
        y="FMS",
        hue="Axis",
        order=S.components(),
        hue_order=S.axes(),
        palette=style.STABILITY_AXIS_COLORS,
        legend=False,
        linewidth=style.LINE_WIDTH_THIN,
        fliersize=1,
        ax=ax,
    )
    common.reference_line(
        ax, MATCH_SCORE_FLOOR, label=f"Weak match ({MATCH_SCORE_FLOOR})"
    )
    ax.set_ylim(0, 1.15)  # legend headroom
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("CP component")
    ax.set_ylabel("CP FMS vs. full cohort")
    ax.legend(loc="upper center")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"], legend=S.legend)
