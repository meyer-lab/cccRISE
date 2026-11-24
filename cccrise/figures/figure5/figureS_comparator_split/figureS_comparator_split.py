"""Supplementary figure (EV5): how the comparators split CCC-RISE's
components. IBD's recovery grids, as Figure 5A-B show ALAD's (A, B); how far
every method's components track detection depth (C); and the co-variation of
every ALAD CCC-RISE component and PB and STAC LR column (D). IBD has no
co-variation panel: the co-variation test was run in ALAD only."""

from ... import common, style
from . import alad_covariation, depth, ibd_best, ibd_containment

LAYOUT = [
    [ibd_best, ibd_containment, depth],
    [alad_covariation],
]


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    # A and B a little down from their letters
    for panel in ("ibd_best", "ibd_containment"):
        common.nudge_panel(fig, panel, up=-2.5 * style.MM)
    return fig
