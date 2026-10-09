"""Supplementary figure: the tensor comparators against each other (no
cccRISE). How much of each comparator column lies in each other
comparator's span (A), and, per pair of comparators, whether the LR pairs
one calls significant (>= 0.4 x a column's largest |loading|) are
significant in the other (B)."""

from .. import common, style
from . import containment, significance

PANELS = {"containment": containment, "significance": significance}
LAYOUT = [["containment"], ["significance"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[0.8, 1.0],
    )
