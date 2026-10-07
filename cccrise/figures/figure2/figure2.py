"""Figure 2: cccRISE uncovers robust, integrative patterns of cell communication."""

from .. import common, style
from . import (
    aggregate_fms,
    celltype_composition,
    cp_factor_matrices,
    cp_rank_selection,
    epithelial_markers,
    nichenet,
    permutation_lr_significance,
    rise_rank_selection,
    sender_receiver_strata,
)

LAYOUT = [
    [rise_rank_selection, cp_rank_selection, aggregate_fms],
    [cp_factor_matrices],
    [sender_receiver_strata, [epithelial_markers, celltype_composition]],
    [permutation_lr_significance, nichenet],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
