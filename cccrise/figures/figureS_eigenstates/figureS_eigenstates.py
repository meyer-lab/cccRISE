"""Supplementary figure: every RISE eigen-state (A) and every cccRISE
component's sender and receiver loading (B) across fine cell types."""

from .. import common, style
from . import canvas, components

PANELS = {"canvas": canvas, "components": components}
LAYOUT = [["canvas"], ["canvas"], ["canvas"], ["components"], ["components"]]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE * 1.3, style.FIGURE_HEIGHT_MAX * 1.5), LAYOUT, PANELS
    )
