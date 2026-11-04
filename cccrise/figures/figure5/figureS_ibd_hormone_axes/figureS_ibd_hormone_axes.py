"""Supplementary figure: the hormone programs' communication axes. Every
eigenstate's cell type association, and among them those C3 and C4 load on
belong to EE, EC and tuft cells: both programs send from EE eigenstates, C3
to an EC eigenstate and C4 to tuft eigenstates. Where Figure 5's zoomed
EE/EC/tuft map sits among all cells, and hormone expression on that map.
Figure 5E-H show the two programs splitting the EE cells expressing their
ligands."""

from ... import common, style
from . import (
    cell_map,
    eigenstate_association,
    eigenstate_composition,
    eigenstate_heatmap,
    eigenstate_loadings,
    hormone_maps,
)

LAYOUT = [
    [eigenstate_heatmap],
    [eigenstate_association, eigenstate_loadings, eigenstate_composition],
    [cell_map],
    [hormone_maps],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
