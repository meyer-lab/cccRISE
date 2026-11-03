"""Supplementary figure: the collinearity of every CP mode across ranks, and
per-component and aggregate RISE and CP stability under patient jackknifing
and seed refits."""

from ... import common, style
from . import cp_collinearity, cpd_fms, rise_fms

LAYOUT = [[cp_collinearity], [rise_fms, cpd_fms]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
