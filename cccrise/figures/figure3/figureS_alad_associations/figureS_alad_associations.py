"""Supplementary figure: association of every CP component with the ALAD
outcomes and the outcome-associated CP component's per-cell sender vs.
receiver weights; association of every RISE component with the outcomes and
the outcome-associated RISE component's condition and gene loadings; and
association of each patient covariate with the outcomes."""

from ... import common
from . import (
    covariates,
    cp_associations,
    rise_associations,
    rise_genes,
    rise_outcome_weights,
    sender_receiver_scatter,
)

LAYOUT = [
    [cp_associations, sender_receiver_scatter],
    [rise_associations],
    [rise_outcome_weights, rise_genes],
    [covariates],
]


def makeFigure():
    return common.compose_fixed(LAYOUT)
