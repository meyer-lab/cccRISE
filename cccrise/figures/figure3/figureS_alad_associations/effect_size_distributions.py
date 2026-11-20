"""Eta-squared of each CPD component's association with the 6-month outcome
across leave-one-patient-out replicates, with the full-cohort value."""

import numpy as np
import seaborn as sns

from ....validation.stability import JACKKNIFE_AXIS
from ... import common, style
from . import _anova_stability as S

AXES = ("square",)


def plot(axs):
    (ax,) = axs
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
    ax.set_ylim(0, S.replicates()["EtaSquared"].max() * 1.4)  # legend headroom
    ax.set_xlabel("CPD component")
    outcome = common.OUTCOME_LABELS.get(S.outcome(), S.outcome())
    ax.set_ylabel(rf"Jackknife $\eta^2$" f"\n({outcome})")
    ax.legend(handles=S.reference_legend_handles(), loc="upper left")
    S.save_source_data()


def makeFigure():
    return common.standalone_fixed(__name__)
