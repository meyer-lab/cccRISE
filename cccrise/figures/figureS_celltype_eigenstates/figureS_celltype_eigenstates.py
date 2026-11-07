"""Supplementary figure: cell type association with every RISE eigenstate,
and each cell type's mean sender and receiver weight on every CP component
(formerly Figure 3's)."""

from .. import common, style
from . import association, celltype_association

LAYOUT = [[association], [celltype_association]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
