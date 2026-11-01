"""Supplementary figure: how much of each cccRISE program's per-cell sender
and receiver weight its (sample, cell type) means explain, against random
splits of the same sizes, coarse and fine labels: whether a pseudobulk could
hold the program at all."""

from ... import common, style
from . import representability_coarse, representability_fine

LAYOUT = [[representability_coarse, representability_fine]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
