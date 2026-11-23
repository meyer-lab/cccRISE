"""Supplementary figure: the hormone programs' communication axes. The
eigenstates C3 and C4 load on belong to EE, EC and tuft cells: both programs
send from EE eigenstates, C3 to an EC eigenstate and C4 to tuft eigenstates.
Where Figure 4's zoomed EE/EC/tuft map sits among all cells, above its
annotated types, beside the programs' receiver weights and their top pairs'
receptors on that map. The magnitude floor behind Figure 4I-K's
hormone classes; inflammation removes villus-like EE cells but not crypt-like
ones; Figure 4K's marker tests across magnitude floors. Figure 4E-H show the
two programs splitting the EE cells expressing their ligands."""

from ... import common, style
from . import (
    cell_map,
    eigenstate_association,
    eigenstate_composition,
    eigenstate_loadings,
    extreme_share,
    magnitude_hist,
    marker_floor_sweep,
    receiver_map_c3,
    receiver_map_c4,
    receptor_maps,
    type_map,
)

LAYOUT = [
    [eigenstate_loadings, eigenstate_association, eigenstate_composition],
    [cell_map, type_map, receiver_map_c3],
    [receiver_map_c4, receptor_maps],
    [magnitude_hist, marker_floor_sweep, extreme_share],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
