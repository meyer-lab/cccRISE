"""Supplementary figure: association of every CP component and every RISE
component with the ALAD patient variables, and the outcome-associated RISE
component's condition and gene loadings."""

from .. import common
from . import (
    cp_associations,
    eigenstate_pacmap,
    rise_associations,
    rise_genes,
    rise_outcome_weights,
)

LAYOUT = [
    [cp_associations],
    [rise_associations],
    [rise_outcome_weights, rise_genes, eigenstate_pacmap],
]


def makeFigure():
    return common.compose_fixed(LAYOUT)
