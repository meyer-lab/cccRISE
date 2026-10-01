"""Supplementary figure: CCC-RISE results across the four communication
scores (factors, outcome association, significance, and stability), and
what each CP component represents under each score."""

from .. import common, style
from . import _shared as S
from . import (
    anova,
    factor_match,
    interpretation,
    significance,
    stability,
    top_k_overlap,
)

PANELS = {
    "factor_match": factor_match,
    "top_k": top_k_overlap,
    "anova": anova,
    "significance": significance,
    "stability": stability,
    "interpretation": interpretation,
}
LAYOUT = [
    ["factor_match"] * 2 + ["top_k"] * 2,
    ["anova"] + ["significance"] * 2 + ["stability"],
    ["interpretation"] * 4,
]
# mm per row; the interpretation grid needs ~30 mm per CP component.
ROW_HEIGHTS = [62, 62, 180]


def makeFigure():
    fig = common.compose(
        (style.FIGURE_WIDTH_DOUBLE, sum(ROW_HEIGHTS) * style.MM),
        LAYOUT,
        PANELS,
        height_ratios=ROW_HEIGHTS,
    )
    S.method_legend(fig)
    return fig
