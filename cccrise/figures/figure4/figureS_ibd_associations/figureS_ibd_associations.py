"""Supplementary figure: association of every IBD CP component with
inflammation status and its contrasts, as Figure 3 and its ALAD supplement
test ALAD's outcomes, beside the loadings of the components associated with
inflammation in CD per group; each biopsy's lineage composition by status;
and each biopsy covariate's association with status."""

from ... import common, style
from . import covariates, cp_associations, lineage_proportions, status_loadings

LAYOUT = [[cp_associations, status_loadings], [lineage_proportions], [covariates]]


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    common.match_row_extent(
        fig, "lineage_proportions", "cp_associations", "status_loadings", True
    )
    return fig
