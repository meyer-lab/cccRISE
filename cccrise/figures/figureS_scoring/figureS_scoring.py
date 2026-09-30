"""Supplementary figure: CCC-RISE results across the four communication
scores (factors, outcome association, significance, and stability)."""

from .. import common, style
from . import _shared as S
from . import anova, factor_match, significance, stability, top_k_overlap

PANELS = {
    "factor_match": factor_match,
    "top_k": top_k_overlap,
    "anova": anova,
    "significance": significance,
    "stability": stability,
}
LAYOUT = [
    ["factor_match"] * 2 + ["top_k"] * 2,
    ["anova"] + ["significance"] * 2 + ["stability"],
]


def makeFigure():
    fig = common.compose((style.FIGURE_WIDTH_DOUBLE, 125 * style.MM), LAYOUT, PANELS)
    S.method_legend(fig)
    return fig
