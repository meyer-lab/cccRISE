"""Supplementary figure: association of every IBD CP component, every RISE
component and each biopsy covariate with inflammation status and its
pairwise contrasts."""

from ... import common, style
from . import covariates, cp_associations, rise_associations

LAYOUT = [[cp_associations, covariates], [rise_associations]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
