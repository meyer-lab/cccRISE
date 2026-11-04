"""Supplementary figure: why the hormone-state resolution holds. Figure 5J's
separation is robust to detection depth and to status; the programs use far
fewer sender eigenstates than the cells they come from; the two hormone
states' markers; and inflammation removes the programs' carrier cells
rather than lowering their per-cell weight."""

from ... import common, style
from . import carrier_decomposition, eigenstate_use, hormone_state_genes, l_call_auroc

LAYOUT = [
    [l_call_auroc, eigenstate_use],
    [hormone_state_genes],
    [carrier_decomposition],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
