"""Figure 3: Alterations in cell communication are associated with ALAD outcomes."""

from .. import common, style
from . import (
    celltype_association,
    cmp_weight_scatter,
    component_weights,
    coordinated_changes,
    pacmaps,
)

LAYOUT = [
    [component_weights, coordinated_changes],
    [celltype_association, cmp_weight_scatter],
    [pacmaps],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
