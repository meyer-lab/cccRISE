"""Supplementary figure: properties of every fit's components behind Figure
4. Effective LR pairs against effective samples (A); each method's input
tensor against sequencing depth, which Tensor-cell2cell's tracks (B); and the
homophilic (C) and top pair's (D) shares of squared LR loading."""

from ... import common, style
from . import breadth, homophily, inputs, top_share

LAYOUT = [[breadth, inputs], [homophily, top_share]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
