"""Supplementary figure (EV5): how the comparators split and merge CCC-RISE's
components. IBD's recovery grids, as Figure 5A-B show ALAD's (A, B); how far
every method's components track detection depth (C); the co-variation of
every ALAD CCC-RISE component and PB and STAC LR column (D); the cosine
between IBD C1 and C5 in each CP mode (E), and C1's (F) and C5's (G) carrying
pairs in the columns related to either, which the comparators share. IBD has
no co-variation panel: the co-variation test was run in ALAD only."""

from ... import common, style
from . import (
    alad_covariation,
    depth,
    ibd_best,
    ibd_c1_columns,
    ibd_c1c5_modes,
    ibd_c5_columns,
    ibd_containment,
)

LAYOUT = [
    [ibd_best, ibd_containment, depth],
    [alad_covariation],
    [ibd_c1c5_modes, ibd_c1_columns, ibd_c5_columns],
]


def makeFigure():
    fig = common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
    # A and B a little down from their letters
    for panel in ("ibd_best", "ibd_containment"):
        common.nudge_panel(fig, panel, up=-2.5 * style.MM)
    return fig
