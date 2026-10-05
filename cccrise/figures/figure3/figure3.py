"""Figure 3: Alterations in cell communication are associated with ALAD outcomes."""

from .. import common, style
from . import (
    celltype_association,
    cmp_weight_scatter,
    component_weights,
    coordinated_changes,
    pacmaps,
    rise_genes,
    rise_outcome_weights,
)

LAYOUT = [
    [rise_outcome_weights, component_weights, cmp_weight_scatter],
    [coordinated_changes, celltype_association],
    [pacmaps, rise_genes],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
