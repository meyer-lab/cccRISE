"""Supplementary figure: association of every CP component and every RISE
component with the ALAD patient variables, the outcome-associated RISE
component's condition and gene loadings, and the outcome-associated CP
component's per-cell sender vs. receiver weights."""

from ... import common
from . import (
    cp_associations,
    rise_associations,
    rise_genes,
    rise_outcome_weights,
    sender_receiver_scatter,
)

LAYOUT = [
    [cp_associations],
    [rise_associations],
    [rise_outcome_weights, rise_genes, sender_receiver_scatter],
]


def makeFigure():
    return common.compose_fixed(LAYOUT)
