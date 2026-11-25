"""Figure 3: Alterations in cell communication are associated with ALAD outcomes."""

import itertools

from .. import common, style
from . import (
    celltype_eigenstates,
    celltype_pacmap,
    coordinated_changes,
    lr_loadings,
    lr_permutation,
    lr_volcano,
    outcome_association,
    sender_receiver_scatter,
)

LAYOUT = [
    [outcome_association, celltype_eigenstates],
    [celltype_pacmap, sender_receiver_scatter],
    [lr_loadings, coordinated_changes],
    [lr_permutation, lr_volcano],
]


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    # Each row raised into the room the row above leaves over it
    names = [[m.__name__.rsplit(".", 1)[-1] for m in row] for row in LAYOUT]
    for upper, lower in itertools.pairwise(names):
        common.close_row_gap(fig, upper, lower)
    return fig
