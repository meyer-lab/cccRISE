"""Supplementary figure: each sample's annotated cell type composition by
ALAD status; association of every CP component with the ALAD outcomes, and
the outcome-associated CP component's per-cell sender vs. receiver weights;
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
    sender_receiver_scatter,
    significant_fraction,
)

LAYOUT = [
    [celltype_proportions],
    [cp_associations],
    [rise_associations],
    [sender_receiver_scatter, covariates],
    [effect_size_distributions, significant_fraction],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
