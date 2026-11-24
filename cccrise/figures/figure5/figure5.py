"""Figure 5: what a cell-type-blind model finds that label-based models do not.
Each ALAD CCC-RISE component's best single LR column in every comparator fit
(A), its containment in each fit's LR span (B), and every component's best
single column against its number of carrying pairs, both cohorts (C); ALAD
C5's carrying pairs in its related comparator columns (D); what each method
keeps and loses when its cells are described more finely, ALAD (E) and IBD
(F); what STAC reports for BTC-EGFR (G) and for PYY and GCG (H) beside what
CCC-RISE resolves; and the fraction NICHES-positive of CCC-RISE's populations
against what cell type predicts (I)."""

from .. import common, style
from . import (
    cos2_grid,
    cos2_pairs,
    flow_alad,
    flow_ibd,
    labels_alad,
    labels_ibd,
    niches_fraction,
    pair_columns,
    span_containment,
)

LAYOUT = [
    [cos2_grid, span_containment, cos2_pairs],
    [pair_columns, labels_alad, labels_ibd],
    [flow_alad, flow_ibd, niches_fraction],
]


#: Room grown over the figure, so the top row sits off the page's top (in)
TOP_ROOM = 2 * style.MM


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    # C, laid out short enough to fit row 1, raised beside A and B's method
    # names and a little right; E and F lower, their cohort bars further under
    # their letters
    common.nudge_panel(fig, "cos2_pairs", right=2 * style.MM, grow=6.5 * style.MM)
    for panel in ("labels_alad", "labels_ibd"):
        common.nudge_panel(fig, panel, up=-3.5 * style.MM)
    common.grow_figure(fig, top=TOP_ROOM)
    return fig
