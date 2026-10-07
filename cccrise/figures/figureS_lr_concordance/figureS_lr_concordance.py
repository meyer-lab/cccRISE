"""Supplementary figure: do cccRISE and the tensor comparators agree on top
LR pairs? Top-10 overlap of every component pair (A); agreement in the head
against the tail (B); for every CP comparator column, agreement on LR pairs,
cell types and samples with its best cccRISE match (C); and pairs one method
weights and the other misses (D)."""

from .. import common, style
from . import head_tail, missed, overlap, who_where

PANELS = {
    "overlap": overlap,
    "head_tail": head_tail,
    "who_where": who_where,
    "missed": missed,
}
LAYOUT = [
    ["overlap", "overlap"],
    ["head_tail", "who_where"],
    ["missed", "missed"],
]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[1.1, 1.1, 0.85],
    )
