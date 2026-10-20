"""Supplementary figure: the comparator programs cccRISE does not contain on
ALAD, by pre-registered class, how concentrated they are, and the candidates
left once one-sample, marker or homophilic, and abundance columns are set
aside."""

from .. import common
from . import candidates, classes, concentration

LAYOUT = [[classes, concentration], [candidates]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
