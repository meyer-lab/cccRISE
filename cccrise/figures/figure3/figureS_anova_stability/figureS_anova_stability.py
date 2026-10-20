"""Supplementary figure: the CP component - outcome associations under patient
jackknifing."""

from ... import common, style
from . import effect_size_distributions, significant_fraction

PANELS = {"distributions": effect_size_distributions, "fraction": significant_fraction}
LAYOUT = [["distributions"] * 2 + ["fraction"]]


def makeFigure():
    return common.compose((style.FIGURE_WIDTH_DOUBLE, 60 * style.MM), LAYOUT, PANELS)
