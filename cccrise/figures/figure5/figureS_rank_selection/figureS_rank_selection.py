"""Supplementary figure: IBD rank selection, Figure 2A-D's for ALAD with the
fit R2X across seeded starts in place of fit and imputation R2X."""

from ... import common, style
from . import cp_collinearity, cp_minima, cp_r2x_starts, rise_rank_selection

LAYOUT = [[rise_rank_selection, cp_r2x_starts, cp_collinearity, cp_minima]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
