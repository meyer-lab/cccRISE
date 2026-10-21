"""Supplementary figure: the masked-imputation ridge leaves the full-data CP
fit unchanged at the selected rank."""

from .. import common
from . import ridge_fms, ridge_r2x

LAYOUT = [[ridge_r2x, ridge_fms]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
