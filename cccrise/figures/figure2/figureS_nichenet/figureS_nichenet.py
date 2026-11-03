"""Supplementary figure: the per-ligand NicheNet target scores and
weight-counted score percentiles behind Figure 2's NicheNet validation, and
the shuffled-prior test of those percentiles."""

from ... import common, style
from . import ligand_scores, prior_shuffle, weighted_percentiles

LAYOUT = [[ligand_scores, weighted_percentiles], [prior_shuffle]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
