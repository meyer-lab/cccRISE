"""Supplementary figure: association of every CP component with the ALAD
outcomes, the loadings of the CP components associated with death, and the
outcome-associated CP component's per-cell sender vs. receiver weights;
association of every RISE component with the outcomes; and association of
each patient covariate with the outcomes."""

from ... import common
from . import (
    covariates,
    cp_associations,
    death_weights,
    rise_associations,
    sender_receiver_scatter,
)

LAYOUT = [
    [cp_associations],
    [death_weights, sender_receiver_scatter],
    [rise_associations],
    [covariates],
]


def makeFigure():
    return common.compose_fixed(LAYOUT)
