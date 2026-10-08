"""Supplementary figure: do cccRISE and the tensor comparators agree on top
LR pairs? How well each comparator fit recovers each cccRISE component (A); for every CP
comparator column, agreement on LR pairs, cell types and samples with its
best cccRISE match (B); and, per comparator, whether the pairs one method
calls significant (>= 0.4 x a column's largest |loading|) are significant in
the other (C)."""

from types import SimpleNamespace

from .. import common, style
from . import recovery, significance, who_where

PANELS = {
    "recovery": recovery,
    "who_where": who_where,
    "significance": SimpleNamespace(plot=significance.plot_row),
}
LAYOUT = [["recovery"], ["who_where"], ["significance"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[0.6, 0.95, 1.15],
    )
