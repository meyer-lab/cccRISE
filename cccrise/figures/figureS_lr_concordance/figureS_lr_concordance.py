"""Supplementary figure: do cccRISE and the tensor comparators agree on LR
programs? How much of each cccRISE LR column lies in each comparator fit's
span (A) and of each comparator column in cccRISE's (B); per comparator fit,
whether the pairs one method calls significant (>= 0.4 x a column's largest
|loading|) are significant in the other (C); and which methods call each
such pair (D)."""

from types import SimpleNamespace

from .. import common, style
from . import containment, membership, significance

PANELS = {
    "containment": containment,
    "containment_reverse": SimpleNamespace(plot=containment.plot_reverse),
    "significance": significance,
    "membership": membership,
}
LAYOUT = [
    ["containment", "containment", "containment_reverse", "containment_reverse"],
    ["significance", "significance", "significance", "membership"],
]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[0.8, 1.0],
    )
