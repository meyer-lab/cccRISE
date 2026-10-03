"""Bootstrap eta-squared interval per CP component, ranked by median and
labeled with the component's top LR pair in the reference fit."""

import numpy as np

from .. import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    table = S.summary().sort_values("MedianEtaSquared").reset_index(drop=True)
    colors = S.reference_colors(table)
    y = np.arange(len(table))
    for yi, component, low, high, median in zip(
        y,
        table["Component"],
        table["CILow"],
        table["CIHigh"],
        table["MedianEtaSquared"],
        strict=True,
    ):
        color = colors[int(component)]
        ax.plot([low, high], [yi, yi], color=color, lw=2)
        ax.scatter(median, yi, color=color, s=style.POINT_SIZE, zorder=3)
    ax.set_yticks(
        y,
        [
            f"C{c}  {pair}"
            for c, pair in zip(table["Component"], table["TopLRPair"], strict=True)
        ],
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.set_xlabel(r"$\eta^2$ (median, bootstrap interval)")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
