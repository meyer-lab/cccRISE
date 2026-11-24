"""Supplementary figure: the ALAD cohort; the collinearity of every CP mode
across ranks; the canonical ALAD RISE factor matrices, sized as Figure 2's CP
factor matrices; per-component RISE and CP stability under patient
jackknifing; and beside them the fraction of each LR pair's entries passing
the cell-permutation test."""

from ... import common, style
from . import (
    cohort,
    cp_collinearity,
    cpd_fms,
    factors,
    genes,
    permutation_fraction,
    rise_fms,
)

LAYOUT = [
    [cohort],
    [cp_collinearity],
    [factors, genes],
    [rise_fms, cpd_fms, permutation_fraction],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
