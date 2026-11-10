"""Figure 5: what a cell-type-blind model finds that label-based models do not.
Each CCC-RISE component's best single comparator column against how many LR
pairs carry it (A); how much of each component's per-cell weight pseudobulk
could hold against what the closest comparator column holds (B); each
component's containment in each fit's LR span (C); ALAD C4's carrying pairs
in its related comparator columns (D) and the co-variation of each (E);
NicheNet on every column of both cohorts (F); what STAC reports for BTC-EGFR
(G) and for PYY and GCG (H) beside what CCC-RISE resolves; and the fraction
NICHES-positive of CCC-RISE's populations against what cell type predicts
(I)."""

from .. import common, style
from . import (
    cos2_pairs,
    flow_alad,
    flow_ibd,
    nichenet,
    niches_fraction,
    pair_columns,
    program_covariation,
    span_containment,
    variance_explained,
)

LAYOUT = [
    [cos2_pairs, variance_explained, span_containment],
    [pair_columns, program_covariation, nichenet],
    [flow_alad, flow_ibd, niches_fraction],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
