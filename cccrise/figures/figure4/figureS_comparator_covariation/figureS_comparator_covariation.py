"""Supplementary figure: Figure 4B and C for every component of every fit,
PB included, each component at its own x, aligned down the panels: how many
top pairs each tests (A), and their co-variation over all pairs of pairs (B)
and over only those sharing no gene (C)."""

from ... import common, style
from . import all_pairs, disjoint, pairs_used

LAYOUT = [[pairs_used], [all_pairs], [disjoint]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
