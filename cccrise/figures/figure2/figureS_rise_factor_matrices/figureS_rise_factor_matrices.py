"""Supplementary figure: the canonical ALAD RISE factor matrices, sized as
Figure 2's CP factor matrices."""

from ... import common, style
from . import factors, genes

LAYOUT = [[factors, genes]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
