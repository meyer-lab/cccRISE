"""Supplementary figure: the patient-permutation test of the outcome-associated
CP component's top LR pairs across set sizes, and its individual pairs."""

from ... import common, style
from . import set_size, volcano

LAYOUT = [[set_size, volcano]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
