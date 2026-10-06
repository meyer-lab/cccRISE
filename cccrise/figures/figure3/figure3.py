"""Figure 3: Alterations in cell communication are associated with ALAD outcomes."""

from .. import common, style
from . import (
    celltype_association,
    coordinated_changes,
    lr_loadings,
    outcome_association,
    pacmaps,
    sender_receiver_scatter,
)

LAYOUT = [
    [outcome_association, celltype_association, sender_receiver_scatter],
    [lr_loadings, coordinated_changes],
    [pacmaps],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
