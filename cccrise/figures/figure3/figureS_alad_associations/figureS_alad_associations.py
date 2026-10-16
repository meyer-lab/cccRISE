"""Supplementary figure: association of every CP component with the ALAD
outcomes, the outcome-associated CP component's per-cell sender vs. receiver
weights, and the loadings of the CP components associated with death;
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
    [cp_associations, sender_receiver_scatter],
    [death_weights],
    [rise_associations],
    [covariates],
]


def makeFigure():
    return common.compose_fixed(LAYOUT)
