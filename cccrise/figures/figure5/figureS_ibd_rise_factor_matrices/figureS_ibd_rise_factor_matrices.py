"""Supplementary figure: the IBD RISE factor matrices, laid out as ALAD's
(`figure2.figureS_rise_factor_matrices`)."""

from ...figure2.figureS_rise_factor_matrices.figureS_rise_factor_matrices import (
    stacked,
)
from . import factors, genes

LAYOUT = [[[factors, genes]]]  # one stack, so the two heatmaps left-align


def makeFigure():
    return stacked(LAYOUT)
