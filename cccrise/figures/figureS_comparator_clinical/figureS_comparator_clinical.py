"""Supplementary figure: clinical association of every method's components
(Figure 3's association heatmap, per method), with the comparators fit on
coarse and on fine cell types, and each method's strongest association under
the permutation null."""

from .. import common, style
from . import heatmaps, heatmaps_fine, maxt

PANELS = {"coarse": heatmaps, "fine": heatmaps_fine, "maxt": maxt}
LAYOUT = [["coarse"], ["fine"], ["maxt"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX), LAYOUT, PANELS
    )
