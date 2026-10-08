"""Supplementary figure: do cccRISE and the tensor comparators agree on top
LR pairs? How much of each cccRISE LR column lies in each comparator fit's span (A);
for every CP comparator column, agreement on LR pairs, cell types and
samples with its best cccRISE match (B); and, per comparator fit, whether
the pairs one method calls significant (>= 0.4 x a column's largest
|loading|) are significant in the other (C)."""

from .. import common, style
from . import containment, significance, who_where

PANELS = {
    "containment": containment,
    "who_where": who_where,
    "significance": significance,
}
LAYOUT = [["containment", "who_where"], ["significance", "significance"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[0.85, 1.0],
    )
