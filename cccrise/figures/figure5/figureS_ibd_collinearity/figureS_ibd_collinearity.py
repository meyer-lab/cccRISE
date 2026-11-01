"""Supplementary figure: IBD CP collinearity of the modes other than LR pair
across seeded starts, as the stability supplement shows for ALAD."""

from ... import common, style
from . import mode_collinearity

LAYOUT = [[mode_collinearity]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
