"""Figure 5: what a cell-type-blind model finds that label-based models do not.
Each CCC-RISE component's best comparator column against its pseudobulk
ceiling (A) and its containment in each fit's LR span (B); ALAD C4's carrying
pairs in its related comparator columns (C) and the co-variation of each (D);
NicheNet on every column of both cohorts (E); what STAC reports for BTC-EGFR
(F) and for PYY and GCG (G) beside what CCC-RISE resolves; and the fraction
NICHES-positive of CCC-RISE's populations against what cell type predicts
(H)."""

from .. import common, style
from . import (
    flow_alad,
    flow_ibd,
    nichenet,
    niches_fraction,
    pair_columns,
    program_covariation,
    program_map,
    span_containment,
)

LAYOUT = [
    [program_map, span_containment],
    [pair_columns, program_covariation, nichenet],
    [flow_alad, flow_ibd, niches_fraction],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
