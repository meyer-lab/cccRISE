"""Supplementary figure: IBD rank selection, Figure 2A-D's for ALAD with BiCV
test on its own axis, the fit R2X across seeded starts in place of fit and
imputation R2X, and the collinearity of the other CP modes."""

from ... import common, style
from . import (
    bicv_test,
    cp_collinearity,
    cp_minima,
    cp_r2x_starts,
    mode_collinearity,
    rise_rank_selection,
)

LAYOUT = [
    [rise_rank_selection, bicv_test, cp_r2x_starts],
    [cp_collinearity, cp_minima, mode_collinearity],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
