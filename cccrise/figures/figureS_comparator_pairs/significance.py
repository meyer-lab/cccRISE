"""Are the pairs one comparator calls significant also significant in
another? As `figureS_lr_concordance.significance`, with a comparator in
cccRISE's place: per pair of comparators (columns) and annotation (rows),
every pair in the top 10 of any column of either fit at (first method's
best |loading| / column max, second's). Points: significant in both
(black), in one only (that method's colour), in neither (small, grey)."""

import pandas as pd
from matplotlib.lines import Line2D

from .. import common, style
from ..figureS_lr_concordance import significance as base
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT

PAIRS = [(METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])]
COLORS = dict(zip(METHODS, style.CATEGORICAL[1:4], strict=True))
# Neutral fills: either axis can be either method.
SHADES = {"cccRISE Only": "0.96", "Shared": "0.9", "Comparator Only": "0.96"}


def _colors(table, x_method, y_method):
    return pd.Series(
        [
            "black" if sx and sy
            else COLORS[x_method] if sx
            else COLORS[y_method] if sy
            else style.COLORS["muted"]
            for sx, sy in zip(table["sx"], table["sy"], strict=True)
        ],
        index=table.index,
    )  # fmt: skip


def _panel(ax, x_method, y_method, annotation):
    table = base.significant(base.scores(y_method, annotation, ref=x_method))
    table = table.assign(c3_flag=False)
    base.shade(ax, c3=False, shades=SHADES)
    base.scatter(ax, table, _colors(table, x_method, y_method))
    ax.set_title(
        f"{SHORT[x_method]} vs {SHORT[y_method]} ({annotation})",
        fontsize=style.FONT_SIZE_TITLE,
    )
    ax.set_xlabel(f"{SHORT[x_method]} |loading| / max")
    ax.set_ylabel(f"{SHORT[y_method]} |loading| / max")


def _legend(fig):
    marker = {"marker": "o", "ls": "", "markersize": 4, "mec": "white", "mew": 0.4}
    handles = [Line2D([], [], color="black", label="both", **marker)] + [
        Line2D([], [], color=COLORS[m], label=f"{SHORT[m]} only", **marker)
        for m in METHODS
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=len(handles),
        fontsize=style.FONT_SIZE_DENSE, frameon=False, handletextpad=0.2,
    )  # fmt: skip


def plot(fig):
    axes = fig.subplots(len(ANNOTATIONS), len(PAIRS), squeeze=False)
    for row, annotation in zip(axes, ANNOTATIONS, strict=True):
        for ax, (x_method, y_method) in zip(row, PAIRS, strict=True):
            _panel(ax, x_method, y_method, annotation)
    _legend(fig)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 120 * style.MM))
