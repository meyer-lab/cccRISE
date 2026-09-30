"""Figure 2: cccRISE uncovers robust, integrative patterns of cell communication."""

from .. import common, style
from . import (
    coordinated_changes,
    cpd_rank,
    nichenet,
    rise_celltype_alignment,
    rise_rank,
    significance,
)

PANELS = {
    "rise_rank": rise_rank,
    "alignment": rise_celltype_alignment,
    "cpd_rank": cpd_rank,
    "significance": significance,
    "nichenet": nichenet,
    "coordinated": coordinated_changes,
}
LAYOUT = [
    ["rise_rank"] * 2 + ["alignment"] * 4,
    ["cpd_rank"] * 6,
    ["significance"] * 6,
    ["nichenet"] * 4 + ["coordinated"] * 2,
]


def makeFigure():
    return common.compose(style.FIGSIZE["full_page"], LAYOUT, PANELS)
