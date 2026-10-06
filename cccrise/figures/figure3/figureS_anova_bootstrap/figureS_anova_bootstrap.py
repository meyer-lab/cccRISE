"""Supplementary figure: patient-resampled bootstrap of the CP component -
outcome associations."""

from ... import common, style
from . import (
    effect_size_distributions,
    effect_size_intervals,
    match_scores,
    significance_stability,
)

PANELS = {
    "distributions": effect_size_distributions,
    "intervals": effect_size_intervals,
    "significance": significance_stability,
    "match": match_scores,
}
LAYOUT = [
    ["distributions", "intervals"],
    ["significance", "match"],
]


def makeFigure():
    return common.compose((style.FIGURE_WIDTH_DOUBLE, 120 * style.MM), LAYOUT, PANELS)
