"""Figure 2: cccRISE uncovers robust, integrative patterns of cell communication."""

from .. import common, style
from . import (
    cmp_eigenstate_loadings,
    cmp_weight_scatter,
    cpd_rank,
    eigenstate_scatter,
    factor_matrices,
    nichenet,
    rise_rank,
    significance,
    stability,
)

LAYOUT = [
    [rise_rank, cpd_rank, stability],
    [factor_matrices],
    [cmp_eigenstate_loadings, eigenstate_scatter, cmp_weight_scatter],
    [significance, nichenet],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
