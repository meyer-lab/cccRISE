"""Supplementary figure: each sample's annotated cell type composition by
ALAD status; association of every CP component with the ALAD outcomes;
association of every RISE component with the outcomes; association of each
patient covariate with the outcomes; and the CP component - outcome
associations under patient jackknifing."""

from ... import common, style
from . import (
    celltype_proportions,
    covariates,
    cp_associations,
    effect_size_distributions,
    rise_associations,
    significant_fraction,
)

LAYOUT = [
    [celltype_proportions],
    [cp_associations],
    [rise_associations],
    [covariates],
    [effect_size_distributions, significant_fraction],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
