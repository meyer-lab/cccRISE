"""Supplementary figure: the granularity component's sender and receiver
loadings on each RISE eigenstate, the canonical cell type markers of
the label-free sender and receiver groups (Figure 2), the genes
distinguishing the groups from each other, and their robustness to the group
cutoff."""

from ... import common, style
from . import (
    celltype_markers,
    cmp_eigenstate_loadings,
    sender_vs_receiver,
    tau_sweep,
)

LAYOUT = [
    [cmp_eigenstate_loadings],
    [celltype_markers],
    [sender_vs_receiver, tau_sweep],
]


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    common.match_row_extent(
        fig, "cmp_eigenstate_loadings", "celltype_markers", "celltype_markers"
    )
    return fig
