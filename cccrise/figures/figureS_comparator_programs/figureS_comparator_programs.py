"""Supplementary figure: how general the components of each method are (A),
and how concentrated the comparators' components are on single cell types
and single samples (B), and the pathways each CP fit's components carry (C)."""

from .. import common, style
from . import concentration, pathways, statistics

PANELS = {
    "statistics": statistics,
    "concentration": concentration,
    "pathways": pathways,
}
LAYOUT = [["statistics", "statistics"], ["concentration", "pathways"]]


def makeFigure():
    return common.compose((style.FIGURE_WIDTH_DOUBLE, 150 * style.MM), LAYOUT, PANELS)
