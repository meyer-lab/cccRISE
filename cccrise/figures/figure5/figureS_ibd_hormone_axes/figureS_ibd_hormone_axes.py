"""Supplementary figure: the hormone programs' communication axes. Every
eigenstate's cell type association, and among them those C3 and C4 load on
belong to EE, EC and tuft cells: both programs send from EE eigenstates, C3
to an EC eigenstate and C4 to tuft eigenstates. Where Figure 5's zoomed
EE/EC/tuft map sits among all cells, beside the programs' receiver weights
and their top pairs' receptors on that map, and the villus-like hormones on
the zoomed map. The programs use far fewer sender eigenstates than the cells
they come from, and the two hormone states' markers. Figure 5E-H show the two
programs splitting the EE cells expressing their ligands."""

from ... import common, style
from . import (
    cell_map,
    eigenstate_association,
    eigenstate_composition,
    eigenstate_heatmap,
    eigenstate_loadings,
    eigenstate_use,
    hormone_maps,
    hormone_state_genes,
    receiver_maps,
    receptor_maps,
)

LAYOUT = [
    [eigenstate_heatmap],
    [eigenstate_loadings, eigenstate_association, eigenstate_composition],
    [cell_map, [receiver_maps], [receptor_maps]],
    [hormone_maps],
    [eigenstate_use],
    [hormone_state_genes],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
