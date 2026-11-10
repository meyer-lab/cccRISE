"""Supplementary figure: how the comparators split and merge CCC-RISE's
components. Each component's best single LR column in every fit (A), to read
against Figure 5B's containment; ALAD C3's carrying pairs in its related
columns of each comparator (B) and their co-variation (C); IBD C1 and C5
together (D, E), which share related columns; and the cosine between C1 and
C5 in each CP mode (F)."""

from ... import common, style
from . import (
    alad_c3_columns,
    alad_c3_covariation,
    cos2_grid,
    ibd_c1c5_columns,
    ibd_c1c5_covariation,
    ibd_c1c5_modes,
)

LAYOUT = [
    [cos2_grid, alad_c3_columns, alad_c3_covariation],
    [ibd_c1c5_columns, ibd_c1c5_covariation, ibd_c1c5_modes],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
