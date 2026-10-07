"""Supplementary figure: clinical association of every method's components
(Figure 3's association heatmap, per method), with the comparators fit on
coarse (A) and on fine (B) cell types, and the patient-resampled bootstrap of
the 6-month outcome association (C)."""

from .. import common, style
from . import bootstrap, heatmaps, heatmaps_fine

PANELS = {
    "coarse": heatmaps,
    "fine": heatmaps_fine,
    "bootstrap": bootstrap,
}
LAYOUT = [["coarse"], ["fine"], ["bootstrap"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        LAYOUT,
        PANELS,
        height_ratios=[1, 1, 0.75],
    )
