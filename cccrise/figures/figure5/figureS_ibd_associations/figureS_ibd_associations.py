"""Supplementary figure: association of every IBD CP component with
inflammation status and its contrasts, as Figure 3 and its ALAD supplement
test ALAD's outcomes; the loadings of the components associated with
inflammation in CD per group; and each biopsy covariate's association with
status."""

from ... import common, style
from . import covariates, cp_associations, status_loadings

LAYOUT = [[cp_associations], [status_loadings], [covariates]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
