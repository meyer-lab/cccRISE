"""Figure 3: Alterations in cell communication are associated with ALAD outcomes."""

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
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
