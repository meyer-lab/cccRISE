"""Per CP component and resampling axis, the fraction of replicates reaching
p < alpha, and -- for a two-level outcome, where a group-mean direction exists
-- agreeing with the full cohort on which group loads higher (black ticks)."""

import numpy as np
from matplotlib.lines import Line2D

from ... import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    axes = S.axes()
    summary = S.summary().set_index(["Axis", "Component"])
    x = np.arange(len(S.components()))
    width = 0.8 / len(axes)
    has_direction = summary["FracSameSignAsReference"].notna().any()
    for axis, offset in zip(axes, S.dodge(len(axes)), strict=True):
        rows = summary.loc[axis].reindex(S.components())
        ax.bar(
            x + offset,
            rows["FracSignificant"],
            width,
            color=style.STABILITY_AXIS_COLORS[axis],
        )
        if has_direction:
            ax.scatter(
                x + offset,
                rows["FracSameSignAsReference"],
                marker="_",
                color="black",
                s=30,
                zorder=3,
            )
    common.reference_line(ax, 0.5)
    ax.set_xticks(x, S.components())
    ax.set_ylim(0, 1.25)  # legend headroom
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("CP component")
    ax.set_ylabel(f"Fraction of replicates, p < {S.alpha():g}")
    if has_direction:
        ax.legend(
            handles=[
                Line2D([], [], marker="_", color="black", lw=0, label="Same direction")
            ],
            loc="upper center",
        )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"], legend=S.legend)
