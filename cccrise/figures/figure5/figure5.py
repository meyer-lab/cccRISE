"""Figure 5: what a cell-type-blind model finds that label-based models do not.
Each cccRISE program between its best comparator column and its label ceiling
(A); what each comparator column is (B); each program's best column in every
fit (C); how STAC arranges the pairs of two mixed programs (D) and whether its
columns keep their co-variation (E); what STAC reports for BTC-EGFR (F) and for
PYY and GCG (G) beside what cccRISE resolves; and the NICHES-positive cells of
cccRISE's populations against what their labels predict (H)."""

from .. import common, style
from . import (
    best_column,
    column_census,
    flow_alad,
    flow_ibd,
    niches_fraction,
    pair_columns,
    program_covariation,
    program_map,
)

LAYOUT = [
    [program_map, column_census],
    [best_column, pair_columns, program_covariation],
    [flow_alad, flow_ibd, niches_fraction],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
