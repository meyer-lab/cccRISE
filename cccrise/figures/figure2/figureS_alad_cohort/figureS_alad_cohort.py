"""Supplementary figure: the ALAD cohort; the canonical ALAD RISE factor
matrices, sized as Figure 2's CP factor matrices; the collinearity of every
CP mode across ranks; and per-component and aggregate RISE and CP stability
under patient jackknifing and seed refits, together as wide as the
collinearity panel."""

from ... import common, style
from . import cohort, cp_collinearity, cpd_fms, factors, genes, rise_fms

LAYOUT = [[cohort], [factors, genes], [cp_collinearity], [rise_fms, cpd_fms]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
