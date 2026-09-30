"""Supplementary figure: RISE and CPD component stability under resampling."""

from .. import common, style
from . import aggregate_fms, match_ambiguity, per_component_fms

PANELS = {
    "per_component": per_component_fms,
    "aggregate": aggregate_fms,
    "ambiguity": match_ambiguity,
}
LAYOUT = [
    ["per_component"] * 2,
    ["aggregate", "ambiguity"],
]


def makeFigure():
    return common.compose((style.FIGURE_WIDTH_DOUBLE, 120 * style.MM), LAYOUT, PANELS)
