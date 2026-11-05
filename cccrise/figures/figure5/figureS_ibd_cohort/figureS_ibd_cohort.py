"""Supplementary figure: the IBD cohort, as ALAD's (`figureS_alad_cohort`):
its schematic placeholder, then its annotated lineage composition by
inflammation status."""

from ... import common, style
from . import cohort, lineage_proportions

LAYOUT = [[cohort], [lineage_proportions]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
