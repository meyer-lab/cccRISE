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
    match_row_extent(fig, "lineage_proportions", "cp_associations", "status_loadings")
    return fig


def match_row_extent(fig, panel: str, first: str, last: str) -> None:
    """Narrow panel module `panel`'s axes so it, with its labels, spans from
    panel `first`'s left extent to panel `last`'s right, letter included."""
    left = _extent(common.panel_axes(fig, first)).x0
    right = _extent([ax for ax in fig.axes if ax.get_label() == last][-1]).x1
    ax = common.panel_axes(fig, panel)
    tight, box = _extent(ax), ax.get_window_extent()
    x0 = left + box.x0 - tight.x0
    x1 = right - (tight.x1 - box.x1)
    position = ax.get_position()
    width = fig.bbox.width
    ax.set_position((x0 / width, position.y0, (x1 - x0) / width, position.height))
    letters = {text.get_gid(): text for text in fig.texts}
    letters[panel].set_x(letters[first].get_position()[0])


def _extent(ax):
    """`ax`'s extent with its labels and legends, in display pixels."""
    tight = ax.get_tightbbox()
    assert tight is not None
    return tight
