"""Supplementary figure: differential communication under the permutation null."""

from .. import common
from . import differential

LAYOUT = [[differential]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
