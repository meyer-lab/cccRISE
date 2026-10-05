"""Supplementary figure: association of every CP component and every RISE
component with the ALAD patient variables."""

from .. import common
from . import cp_associations, rise_associations

LAYOUT = [[cp_associations], [rise_associations]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
