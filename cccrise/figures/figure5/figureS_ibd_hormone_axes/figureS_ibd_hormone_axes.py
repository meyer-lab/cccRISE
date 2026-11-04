"""Supplementary figure: the hormone programs' communication axes. Every
eigenstate's cell type association, and among them those C3 and C4 load on
belong to EE, EC and tuft cells: both programs send from EE eigenstates, C3
to an EC eigenstate and C4 to tuft eigenstates. Figure 5E-I show the two
programs splitting the EE cells expressing their ligands."""

from ... import common, style
from . import (
    eigenstate_association,
    eigenstate_composition,
    eigenstate_heatmap,
    eigenstate_loadings,
)

LAYOUT = [
    [eigenstate_heatmap],
    [eigenstate_association, eigenstate_loadings, eigenstate_composition],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
