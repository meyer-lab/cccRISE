"""Leave-one-patient-out eta-squared distribution per CP component, with the
full-cohort value overlaid (filled where significant)."""

import numpy as np
import seaborn as sns

from ....validation.stability import JACKKNIFE_AXIS
from ... import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    sns.violinplot(
        data=S.replicates(),
        x="Component",
        y="EtaSquared",
        order=S.components(),
        color=style.STABILITY_AXIS_COLORS[JACKKNIFE_AXIS],
        inner=None,
        linewidth=style.LINE_WIDTH_THIN,
        linecolor=style.COLORS["ink"],
        cut=0,
        density_norm="width",
        ax=ax,
    )
    reference = S.reference()
    S.reference_markers(ax, np.arange(len(reference)), reference["EtaSquared"])
    ax.set_ylim(0, S.replicates()["EtaSquared"].max() * 1.2)  # legend headroom
    ax.set_xlabel("CP component")
    outcome = common.OUTCOME_LABELS.get(S.outcome(), S.outcome())
    ax.set_ylabel(rf"Jackknife $\eta^2$ ({outcome})")
    ax.legend(handles=S.reference_legend_handles(), loc="upper center", ncols=2)
    S.save_source_data()


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
