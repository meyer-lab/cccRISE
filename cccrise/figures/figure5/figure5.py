"""Figure 5: what a cell-type-blind model finds that label-based models do not.
Each ALAD CCC-RISE component's best single LR column in every comparator fit
(A) and its containment in each fit's LR span (B); ALAD C4's carrying pairs
in its related comparator columns (C) and the co-variation of each (D); what
each method keeps and loses when its cells are described more finely, ALAD
(E) and IBD (F); NicheNet on every PB and STAC column and CCC-RISE component
of both cohorts (G); what STAC
reports for BTC-EGFR (H) and for PYY and GCG (I) beside what CCC-RISE
resolves; and the fraction NICHES-positive of CCC-RISE's populations against
what cell type predicts (J)."""

from .. import common, style
from . import (
    cos2_grid,
    flow_alad,
    flow_ibd,
    labels_alad,
    labels_ibd,
    nichenet,
    niches_fraction,
    pair_columns,
    program_covariation,
    span_containment,
)

LAYOUT = [
    [cos2_grid, span_containment, pair_columns, program_covariation],
    [labels_alad, labels_ibd, nichenet],
    [flow_alad, flow_ibd, niches_fraction],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
