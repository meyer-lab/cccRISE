"""Supplementary figure: association of every CP component with every ALAD
outcome."""

from .. import common
from . import associations

LAYOUT = [[associations]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
