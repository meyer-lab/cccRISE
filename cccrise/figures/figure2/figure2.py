"""Figure 2: cccRISE uncovers robust, integrative patterns of cell communication."""

from .. import common, style
from . import (
    composition,
    cpd_rank,
    factor_matrices,
    group_scatter,
    identity_markers,
    nichenet,
    rise_rank,
    significance,
    stability,
)

LAYOUT = [
    [rise_rank, cpd_rank, stability],
    [factor_matrices],
    [group_scatter, [identity_markers, composition]],
    [significance, nichenet],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
