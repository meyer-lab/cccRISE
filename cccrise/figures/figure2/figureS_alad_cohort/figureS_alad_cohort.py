"""Supplementary figure: the ALAD cohort and its annotated cell type
composition by ALAD status."""

from ... import common, style
from . import celltype_proportions, cohort

LAYOUT = [[cohort], [celltype_proportions]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
