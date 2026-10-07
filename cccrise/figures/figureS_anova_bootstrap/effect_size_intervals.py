"""Eta-squared interval per CP component on every resampling axis -- the
percentile interval on the bootstrap axes, the normal interval from the
jackknife standard error on the jackknife axis -- with the replicates' median,
labeled with the component's top LR pair in the reference fit."""

import numpy as np

from .. import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    axes = S.axes()
    reference = S.reference()
    summary = S.summary().set_index(["Axis", "Component"])
    y = np.arange(len(reference))
    offsets = S.dodge(len(axes))
    for axis, offset in zip(axes, offsets, strict=True):
        color = style.STABILITY_AXIS_COLORS[axis]
        rows = summary.loc[axis].reindex(reference["Component"])
        yi = y + offset
        ax.hlines(yi, rows["CILow"], rows["CIHigh"], color=color, lw=1.5)
        ax.scatter(rows["MedianEtaSquared"], yi, color=color, s=4, zorder=3)
    S.reference_markers(ax, reference["EtaSquared"], y)
    ax.set_yticks(
        y,
        [
            f"C{c}  {pair}"
            for c, pair in zip(
                reference["Component"], reference["TopLRPair"], strict=True
            )
        ],
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"$\eta^2$ (median and interval)")
    ax.set_title("Effect size intervals")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"], legend=S.legend)
