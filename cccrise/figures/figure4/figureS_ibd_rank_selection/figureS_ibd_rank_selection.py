"""Supplementary figure: the IBD cohort's schematic placeholder; IBD rank
selection, Figure 2A-D's for ALAD with BiCV test on its own axis, the fit
R2X across seeded starts in place of fit and imputation R2X, and the
collinearity of the other CP modes; the RISE condition and gene factors at
the selected rank, as ALAD's are drawn (`figure2.figureS_alad_cohort`);
and every eigenstate's cell type association."""

from ... import common, style
from . import (
    bicv_test,
    cohort,
    cp_collinearity,
    cp_minima,
    cp_r2x_starts,
    eigenstate_heatmap,
    factors,
    genes,
    mode_collinearity,
    rise_rank_selection,
)

LAYOUT = [
    [cohort],
    [rise_rank_selection, bicv_test, cp_r2x_starts],
    [cp_collinearity, cp_minima, mode_collinearity],
    [factors, genes],
    [eigenstate_heatmap],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
