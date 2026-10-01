"""Supplementary figure: per-component RISE and CPD stability under
resampling, and how unambiguously components are re-identified."""

from .. import common, style
from . import cpd_fms, match_ambiguity, rise_fms

PANELS = {"rise": rise_fms, "cpd": cpd_fms, "ambiguity": match_ambiguity}
LAYOUT = [
    ["rise"] * 2,
    ["cpd", "ambiguity"],
]


def makeFigure():
    fig = common.compose((style.FIGURE_WIDTH_DOUBLE, 130 * style.MM), LAYOUT, PANELS)
    common.stability_legend(fig)
    return fig
