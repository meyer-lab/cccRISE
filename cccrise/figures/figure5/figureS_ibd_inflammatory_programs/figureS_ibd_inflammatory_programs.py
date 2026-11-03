"""Supplementary figure: the inflammatory programs C1 and C5. Every top LR
pair is higher in inflamed than healthy biopsies (Figure 3F's test), and
what the myeloid cells highest in each program's weight express."""

from ... import common, style
from . import high_sender_genes, top_pair_permutation

LAYOUT = [[top_pair_permutation], [high_sender_genes]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
