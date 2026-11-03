"""Supplementary figure: properties of every fit's components behind Figure
4. How each comparator's LR columns match from coarse to fine labels (A);
effective LR pairs against effective samples (B); each method's input
tensor against sequencing depth, which Tensor-cell2cell's tracks (C); and
the homophilic (D) and top pair's (E) shares of squared LR loading behind
Figure 4F's classes."""

from ... import common, style
from . import breadth, coarse_to_fine, homophily, inputs, top_share

LAYOUT = [[coarse_to_fine, breadth], [inputs, homophily], [top_share]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
