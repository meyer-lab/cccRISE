"""Figure 2: cccRISE uncovers robust, integrative patterns of cell communication."""

from .. import common, style
from . import (
    cmp_eigenstate_loadings,
    cmp_weight_scatter,
    coordinated_changes,
    cpd_rank,
    eigenstate_scatter,
    nichenet,
    rise_rank,
    significance,
    stability,
)

PANELS = {
    "rise_rank": rise_rank,
    "cpd_rank": cpd_rank,
    "loadings": cmp_eigenstate_loadings,
    "weights": cmp_weight_scatter,
    "eigenstates": eigenstate_scatter,
    "stability": stability,
    "significance": significance,
    "nichenet": nichenet,
    "coordinated": coordinated_changes,
}
LAYOUT = [
    ["rise_rank"] * 2 + ["cpd_rank"] * 4,
    ["stability"] * 2 + ["coordinated"] * 4,
    ["loadings"] * 2 + ["weights"] * 2 + ["eigenstates"] * 2,
    ["significance"] * 6,
    ["nichenet"] * 6,
]
# mm per row, summing to the full-page height.
ROW_HEIGHTS = [40, 34, 44, 38, 44]


def makeFigure():
    return common.compose(
        style.FIGSIZE["full_page"], LAYOUT, PANELS, height_ratios=ROW_HEIGHTS
    )
