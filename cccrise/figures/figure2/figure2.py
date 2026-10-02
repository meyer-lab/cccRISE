"""Figure 2: cccRISE uncovers robust, integrative patterns of cell communication."""

from .. import common, style
from . import (
    coordinated_changes,
    cpd_rank,
    nichenet,
    rise_celltype_alignment,
    rise_rank,
    significance,
    stability,
)

PANELS = {
    "rise_rank": rise_rank,
    "alignment": rise_celltype_alignment,
    "cpd_rank": cpd_rank,
    "stability": stability,
    "significance": significance,
    "nichenet": nichenet,
    "coordinated": coordinated_changes,
}
LAYOUT = [
    ["rise_rank"] * 2 + ["alignment"] * 4,
    ["cpd_rank"] * 4 + ["stability"] * 2,
    ["significance"] * 6,
    ["nichenet"] * 6,
    ["coordinated"] * 6,
]
# mm per row, summing to the full-page height.
ROW_HEIGHTS = [42, 42, 42, 46, 28]


def makeFigure():
    return common.compose(
        style.FIGSIZE["full_page"], LAYOUT, PANELS, height_ratios=ROW_HEIGHTS
    )
