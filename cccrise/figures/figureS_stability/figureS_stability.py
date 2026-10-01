"""Supplementary figure: per-component RISE and CPD stability under
resampling."""

from .. import common, style
from . import cpd_fms, rise_fms

PANELS = {"rise": rise_fms, "cpd": cpd_fms}
# 3:1 matches the RISE:CPD component counts, so boxes share a width.
LAYOUT = [["rise"] * 3 + ["cpd"]]


def makeFigure():
    fig = common.compose((style.FIGURE_WIDTH_DOUBLE, 75 * style.MM), LAYOUT, PANELS)
    common.stability_legend(fig)
    return fig
