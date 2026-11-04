"""Supplementary figure: association of every IBD CP and RISE component
with inflammation status and its contrasts, as Figure 3 and its ALAD
supplement test ALAD's outcomes; the loadings of the status-associated CP
components per group; and each biopsy covariate's association with status."""

from ... import common, style
from . import covariates, cp_associations, rise_associations, status_loadings

LAYOUT = [[cp_associations], [rise_associations], [status_loadings], [covariates]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
