"""Supplementary figure: do cccRISE and the tensor comparators agree on top
LR pairs? Top-10 overlap of every component pair (A); for every CP
comparator column, agreement on LR pairs, cell types and samples with its
best cccRISE match (B); and, per comparator, whether the pairs one method
calls significant (>= 0.4 x a column's largest |loading|) are significant in
the other (C)."""

from types import SimpleNamespace

from .. import common, style
from . import overlap, significance, who_where

PANELS = {
    "overlap": overlap,
    "who_where": who_where,
    "significance": SimpleNamespace(plot=significance.plot_row),
}
LAYOUT = [["overlap"], ["who_where"], ["significance"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[1.0, 0.85, 1.15],
    )
