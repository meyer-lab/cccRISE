"""Eta-squared distribution per CP component on every resampling axis, with
the full-cohort value overlaid (filled where significant)."""

import numpy as np
import seaborn as sns

from ... import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    axes = S.axes()
    sns.violinplot(
        data=S.replicates(),
        x="Component",
        y="EtaSquared",
        hue="Axis",
        order=S.components(),
        hue_order=axes,
        palette=style.STABILITY_AXIS_COLORS,
        legend=False,
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
    ax.set_ylabel(rf"$\eta^2$ ({S.outcome()})")
    ax.legend(handles=S.reference_legend_handles(), loc="upper center", ncols=2)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"], legend=S.legend)
