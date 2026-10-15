"""Supplementary figure: the per-ligand NicheNet target scores and
weight-counted score percentiles behind Figure 2's NicheNet validation."""

from ... import common, style
from . import ligand_scores, weighted_percentiles

LAYOUT = [[ligand_scores, weighted_percentiles]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
