"""Figure 2: cccRISE uncovers robust, integrative patterns of cell communication."""

import matplotlib.pyplot as plt

from .. import common, style
from . import (
    celltype_composition,
    cp_factor_matrices,
    cp_minima,
    cp_rank_selection,
    epithelial_markers,
    nichenet,
    permutation_lr_significance,
    rise_rank_selection,
    sender_receiver_strata,
)

LAYOUT = [
    [rise_rank_selection, cp_rank_selection, cp_minima],
    [cp_factor_matrices],
    [permutation_lr_significance, nichenet],
    [sender_receiver_strata, [epithelial_markers, celltype_composition]],
]


def makeFigure():
    box = {"axes.edgecolor": style.BOX_COLOR, "axes.linewidth": style.BOX_LINE_WIDTH}
    # Grey tick marks; their labels stay black
    for axis in "xy":
        box |= {f"{axis}tick.color": style.BOX_COLOR, f"{axis}tick.labelcolor": "black"}
    with plt.rc_context(box):
        return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
