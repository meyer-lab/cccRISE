"""Supplementary figure: Figure 4B and C for every component of every fit,
PB included, each component at its own x: top-pair co-variation over all
pairs of pairs (A) and over only those sharing no gene (B)."""

from ... import common, style
from . import all_pairs, disjoint

LAYOUT = [[all_pairs], [disjoint]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
