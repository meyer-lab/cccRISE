"""Supplementary figure, IBD: as `figureS_comparator_missed_alad`."""

from ... import common, style
from . import associations, contents, share

LAYOUT = [[share, associations], [contents]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
