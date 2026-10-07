"""Bootstrap eta-squared distribution per CP component, with the full-cohort
value overlaid; components significant in the full cohort are emphasized."""

import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from ... import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    table = S.summary()
    order = table["Component"].astype(int).tolist()
    sns.violinplot(
        data=S.replicates(),
        x="Component",
        y="EtaSquared",
        hue="Component",
        order=order,
        hue_order=order,
        palette=S.reference_colors(table),
        legend=False,
        inner="quartile",
        linewidth=style.LINE_WIDTH_THIN,
        linecolor=style.COLORS["ink"],
        cut=0,
        density_norm="width",
        ax=ax,
    )
    ax.scatter(
        np.arange(len(order)),
        table["ReferenceEtaSquared"],
        marker="D",
        s=style.POINT_SIZE,
        color="black",
        zorder=5,
    )
    ax.set_ylim(0, S.replicates()["EtaSquared"].max() * 1.25)  # legend headroom
    ax.set_xlabel("CP component")
    ax.set_ylabel(rf"Bootstrap $\eta^2$ ({S.outcome()})")
    ax.legend(
        handles=[
            Line2D([], [], color=style.COLORS["emphasis"], lw=4, label="Ref. p < 0.05"),
            Line2D([], [], color=style.COLORS["muted"], lw=4, label="Ref. n.s."),
            Line2D([], [], marker="D", color="black", lw=0, label="Full cohort"),
        ],
        loc="upper center",
        ncols=3,
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
