"""Supplementary figure: IBD rank selection, Figure 2A-D's for ALAD with BiCV
test on its own axis, the fit R2X across seeded starts in place of fit and
imputation R2X, and the collinearity of the other CP modes; below, the RISE
condition and gene factors at the selected rank, as ALAD's are drawn
(`figure2.figureS_rise_factor_matrices`)."""

from ... import style
from ...figure2.figureS_rise_factor_matrices.figureS_rise_factor_matrices import (
    stacked,
)
from . import (
    bicv_test,
    cp_collinearity,
    cp_minima,
    cp_r2x_starts,
    factors,
    genes,
    mode_collinearity,
    rise_rank_selection,
)

LAYOUT = [
    [rise_rank_selection, bicv_test, cp_r2x_starts],
    [cp_collinearity, cp_minima, mode_collinearity],
    [[factors, genes]],  # one stack, so the two heatmaps left-align
]


def makeFigure():
    return stacked(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
