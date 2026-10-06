"""Figure 3: Alterations in cell communication are associated with ALAD outcomes."""

from .. import common, style
from . import (
    celltype_association,
    cmp_weight_scatter,
    component_weights,
    coordinated_changes,
    lr_loadings,
    pacmaps,
)

LAYOUT = [
    [component_weights, celltype_association, cmp_weight_scatter],
    [lr_loadings, coordinated_changes],
    [pacmaps],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
