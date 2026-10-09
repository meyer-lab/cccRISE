"""Supplementary figure: what does a finer cell-type annotation buy the
comparators? Each coarse type is split by its own expression into 13, 23 and
38 types (against random splits of the same sizes, and the 26-type fine
annotation): agreement with cccRISE (A), containment of C2-C4 (B), how much
of each cccRISE component a (sample x type) mean represents (C), whether a
comparator finds C4 or its clinical association (D), and, where a comparator
column and cccRISE agree on LR pairs, how many cells of the column's cell
types carry the cccRISE program (E)."""

from .. import common, style
from . import (
    cells,
    ladder_cka,
    ladder_clinical,
    ladder_containment,
    representability,
)

PANELS = {
    "cka": ladder_cka,
    "containment": ladder_containment,
    "representability": representability,
    "clinical": ladder_clinical,
    "cells": cells,
}
LAYOUT = [
    ["cka", "containment"],
    ["representability", "clinical"],
    ["cells", "cells"],
]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[1, 1, 0.8],
    )
