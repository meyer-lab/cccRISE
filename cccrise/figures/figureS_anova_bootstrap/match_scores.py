"""How well each replicate's CP component matched the full-cohort component,
under the patient bootstrap and the leave-one-patient-out jackknife -- the
control for the effect-size panels: a wide effect-size distribution with low
match scores means the component was not re-identified, not that its
association was unstable."""

import numpy as np
import seaborn as sns

from ...validation.stability import JACKKNIFE_AXIS, STAGE_CPD
from .. import common, style
from . import _common as S

#: Match score below which a replicate's component is not the reference one
#: in any useful sense.
MATCH_SCORE_FLOOR = 0.7


def plot(fig):
    ax = fig.subplots()
    df = S.stability()
    axes = [a for a in (S.AXIS, JACKKNIFE_AXIS) if a in set(df["Axis"])]
    sns.boxplot(
        data=df[(df["Stage"] == STAGE_CPD) & df["Axis"].isin(axes)],
        x="Component",
        y="FMS",
        hue="Axis",
        order=S.summary()["Component"].astype(int).tolist(),
        hue_order=axes,
        palette=style.STABILITY_AXIS_COLORS,
        linewidth=style.LINE_WIDTH_THIN,
        fliersize=1,
        ax=ax,
    )
    common.reference_line(
        ax, MATCH_SCORE_FLOOR, label=f"Weak match ({MATCH_SCORE_FLOOR})"
    )
    ax.set_ylim(0, 1.25)  # legend headroom
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("CP component")
    ax.set_ylabel("CPD FMS vs. full cohort")
    ax.set_title("Component identifiability")
    ax.legend(loc="upper center", ncols=3)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
