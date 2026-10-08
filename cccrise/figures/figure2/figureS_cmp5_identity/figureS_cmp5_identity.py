"""Supplementary figure: how the granularity component's sender and receiver
weights relate to the RISE eigenstates, the genes distinguishing the
label-free sender and receiver groups (Figure 2) from each other and from the
rest of the cells, and which cell types they resemble."""

from ... import common, style
from . import (
    celltype_auroc,
    celltype_markers,
    cmp_eigenstate_loadings,
    eigenstate_scatter,
    eigenstate_split,
    group_vs_rest,
    sender_vs_receiver,
)

LAYOUT = [
    [eigenstate_scatter, cmp_eigenstate_loadings, eigenstate_split],
    [sender_vs_receiver, group_vs_rest],
    [[celltype_markers, celltype_auroc]],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
