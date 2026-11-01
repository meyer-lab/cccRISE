"""Supplementary figure: containment in both directions on ALAD, coarse
labels (left) and fine (right): each cccRISE program's sample columns in each
comparator (Figure 4B's LR containment, on the sample mode), each comparator
LR column in cccRISE (colored by audit class), and mean LR containment
between every pair of methods."""

from ... import common, style
from . import (
    outside_coarse,
    outside_fine,
    pairwise_coarse,
    pairwise_fine,
    sample_reach_coarse,
    sample_reach_fine,
)

LAYOUT = [
    [sample_reach_coarse, sample_reach_fine],
    [outside_coarse, outside_fine],
    [pairwise_coarse, pairwise_fine],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
