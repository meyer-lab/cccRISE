"""Supplementary figure: the IBD cohort, as ALAD's (`figureS_alad_cohort`):
its schematic placeholder, then its annotated lineage and fine cell type
composition by inflammation status."""

from ... import common, style
from . import cohort, fine_proportions, lineage_proportions

LAYOUT = [[cohort], [lineage_proportions], [fine_proportions]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
