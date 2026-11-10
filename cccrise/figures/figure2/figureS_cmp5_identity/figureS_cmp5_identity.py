"""Supplementary figure: the RISE eigenstates behind the granularity
component's sender and receiver weights, the canonical cell type markers of
the label-free sender and receiver groups (Figure 2), the genes
distinguishing each group from every other cell and from each other, and
their robustness to the group cutoff."""

from ... import common, style
from . import (
    celltype_markers,
    cmp_eigenstate_loadings,
    eigenstate_scatter,
    group_vs_rest,
    sender_vs_receiver,
    tau_sweep,
)

LAYOUT = [
    [eigenstate_scatter, cmp_eigenstate_loadings],
    [celltype_markers],
    [group_vs_rest, sender_vs_receiver],
    [tau_sweep],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
