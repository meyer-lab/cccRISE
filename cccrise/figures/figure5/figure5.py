"""Figure 5: cccRISE finds cohort-wide communication programs that stratify
Crohn's disease biopsies (IBDverse terminal ileum)."""

from .. import common, style
from . import (
    carrier_expression,
    celltype_weights,
    cohort,
    cp_factor_matrices,
    effective_samples,
    group_loadings,
)

LAYOUT = [
    [cohort],
    [cp_factor_matrices],
    [effective_samples, celltype_weights],
    [group_loadings],
    [carrier_expression],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
