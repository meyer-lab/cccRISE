"""Figure 3: Alterations in cell communication are associated with ALAD outcomes."""

from .. import common, style
from . import associations, component_weights, interpretation

PANELS = {
    "associations": associations,
    "weights": component_weights,
    "interpretation": interpretation,
}
LAYOUT = [
    ["associations"] * 3 + ["weights"] * 3,
    ["interpretation"] * 6,
]
# mm per row, summing to the full-page height.
ROW_HEIGHTS = [80, 120]


def makeFigure():
    return common.compose(
        style.FIGSIZE["full_page"], LAYOUT, PANELS, height_ratios=ROW_HEIGHTS
    )
