"""Supplementary figure: the CP component - outcome associations under every
stability resampling axis."""

from .. import common, style
from . import _common as S
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
    ["distributions", "distributions"],
    ["intervals", "significance"],
    ["match", "match"],
]


def makeFigure():
    fig = common.compose((style.FIGURE_WIDTH_DOUBLE, 185 * style.MM), LAYOUT, PANELS)
    S.legend(fig)
    return fig
