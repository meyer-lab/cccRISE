"""Figure 5: cccRISE finds cohort-wide communication programs that stratify
Crohn's disease biopsies (IBDverse terminal ileum)."""

from .. import common, style
from . import cohort, cp_factor_matrices

LAYOUT = [
    [cohort],
    [cp_factor_matrices],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
