"""Supplementary figure: each annotation-pooled comparator's own rank
selection on ALAD, coarse labels first, then fine. Pseudobulk CPD by
cccRISE's own procedure (fit R2X across ten seeded SVD starts with the
Kneedle elbow; as Figure 2C and D, the LR pair mode's collinearity per start
and the distinct minima the starts reach; its chosen rank marked);
Tensor-cell2cell by its authors' elbow, then how nearly two of a start's
components duplicate each other, per start (its non-negative factors overlap
by construction, so per-mode collinearity would not read as pseudobulk
CPD's), and the distinct minima of its ten random starts; STACCato by its
eigenvalue shares on each labelling."""

from ... import common, style
from . import (
    pb_minima_coarse,
    pb_minima_fine,
    pb_rank_selection_coarse,
    pb_rank_selection_fine,
    stac_shares_coarse,
    stac_shares_fine,
    tc2c_minima_coarse,
    tc2c_minima_fine,
    tc2c_rank_selection_coarse,
    tc2c_rank_selection_fine,
)

LAYOUT = [
    [pb_rank_selection_coarse, pb_minima_coarse],
    [tc2c_rank_selection_coarse, tc2c_minima_coarse],
    [pb_rank_selection_fine, pb_minima_fine],
    [tc2c_rank_selection_fine, tc2c_minima_fine],
    [stac_shares_coarse, stac_shares_fine],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
