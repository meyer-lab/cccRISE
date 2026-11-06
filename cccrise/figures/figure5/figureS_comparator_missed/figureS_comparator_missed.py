"""Supplementary figure: what the comparators find that no cccRISE program carries.
The LR contents of those columns, with each pair's place in cccRISE (A), and their
strongest associations on ALAD (B) and IBD (C)."""

from ... import common, style
from . import alad_associations, column_contents, ibd_associations

LAYOUT = [[column_contents], [alad_associations, ibd_associations]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
