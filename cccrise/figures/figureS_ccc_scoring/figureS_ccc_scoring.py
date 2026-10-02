"""Supplementary figure: CCC-RISE across the four communication scores --
how closely each alternate score's CP components match the default's, their
outcome associations, and what each component represents under each score."""

from .. import common, style
from . import _common as S
from . import anova, factor_match, interpretation

PANELS = {
    "factor_match": factor_match,
    "anova": anova,
    "interpretation": interpretation,
}
LAYOUT = [
    ["factor_match"] * 2 + ["anova"],
    ["interpretation"] * 3,
]
# mm per row; the interpretation grid needs ~30 mm per CP component.
ROW_HEIGHTS = [60, 180]


def makeFigure():
    fig = common.compose(
        (style.FIGURE_WIDTH_DOUBLE, sum(ROW_HEIGHTS) * style.MM),
        LAYOUT,
        PANELS,
        height_ratios=ROW_HEIGHTS,
    )
    S.method_legend(fig)
    return fig
