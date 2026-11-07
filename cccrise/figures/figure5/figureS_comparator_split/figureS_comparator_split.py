"""Supplementary figure: depth tracking of every method's components (A), and
IBD's mixed programs: their pairs in STAC's columns (B) and their co-variation
against those columns (C)."""

from ... import common, style
from . import depth_tracking, mixed_covariation, mixed_pairs

LAYOUT = [[depth_tracking], [mixed_pairs, mixed_covariation]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
