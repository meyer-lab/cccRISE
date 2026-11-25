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
    [rise_rank_selection, cp_rank_selection],
    [cp_factor_matrices],
    [aggregate_fms, permutation_lr_significance, nichenet],
    [sender_receiver_strata, [epithelial_markers, celltype_composition]],
]


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    row3 = ["aggregate_fms", "permutation_lr_significance", "nichenet"]
    row4 = ["sender_receiver_strata", "epithelial_markers"]
    common.close_row_gap(fig, row3, row4)
    # M's axes (and N with its letter) raised to a panel gap under row 3,
    # beside L's taller legend, M's letter left at the row's top; then N to a
    # panel gap under M's gene names
    room, _ = common.panel_room(fig, row3, ["epithelial_markers"], lower_letters=False)
    up = room - style.PANEL_GAP_MIN
    common.nudge_panel(fig, "epithelial_markers", up=up)
    common.nudge_panel(fig, "celltype_composition", up=up, letter=True)
    room, _ = common.panel_room(fig, ["epithelial_markers"], ["celltype_composition"])
    common.nudge_panel(
        fig, "celltype_composition", up=room - style.PANEL_GAP_MIN, letter=True
    )
    return fig
