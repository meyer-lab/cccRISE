"""Supplementary figure: the canonical ALAD RISE factor matrices."""

from .. import common
from . import factors, genes

LAYOUT = [[factors], [genes]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
