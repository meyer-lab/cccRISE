"""Supplementary figure (EV5): how the comparators split and merge CCC-RISE's
components. Each component's best single comparator LR column against how
many LR pairs carry it (A); IBD's recovery grids, as Figure 5A-B show
ALAD's (B, C); how far every method's components track detection depth (D);
ALAD C3's carrying pairs in its related columns of each comparator (E) and
their co-variation (F); the cosine between IBD C1 and C5 in each CP mode
(G), and C1's (H) and C5's (I) carrying pairs in the columns related to
either, which the comparators share. IBD has no co-variation panel: the co-variation test was run in
ALAD only."""

from ... import common, style
from . import (
    alad_c3_columns,
    alad_c3_covariation,
    cos2_pairs,
    depth,
    ibd_best,
    ibd_c1_columns,
    ibd_c1c5_modes,
    ibd_c5_columns,
    ibd_containment,
)

LAYOUT = [
    [cos2_pairs, ibd_best, ibd_containment],
    [depth, alad_c3_columns, alad_c3_covariation],
    [ibd_c1c5_modes, ibd_c1_columns, ibd_c5_columns],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
