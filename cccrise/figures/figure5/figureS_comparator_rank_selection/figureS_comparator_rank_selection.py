"""Supplementary figure: each annotation-pooled comparator's own rank
selection, on ALAD and IBD together, one column per method and labelling.
Pseudobulk CPD by cccRISE's rank-selection sweep (fit R2X across ten seeded
SVD starts; as Figure 2C and D, the LR pair mode's collinearity per start and
the share of starts at the most common minimum); Tensor-cell2cell by the
Kneedle elbow of its error, then how nearly two of a start's
components duplicate each other, per start (its non-negative factors overlap
by construction, so per-mode collinearity would not read as pseudobulk
CPD's), and the same share of its ten random starts; STACCato by its
eigenvalue shares on each labelling. Each cohort's chosen rank is marked."""

from ... import common, style
from . import (
    pb_collinearity_coarse,
    pb_collinearity_fine,
    pb_fit_coarse,
    pb_fit_fine,
    pb_minima_coarse,
    pb_minima_fine,
    stac_shares_coarse,
    stac_shares_fine,
    tc2c_congruence_coarse,
    tc2c_congruence_fine,
    tc2c_elbow_coarse,
    tc2c_elbow_fine,
    tc2c_minima_coarse,
    tc2c_minima_fine,
)

LAYOUT = [
    [pb_fit_coarse, pb_fit_fine, tc2c_elbow_coarse, tc2c_elbow_fine],
    [
        pb_collinearity_coarse,
        pb_collinearity_fine,
        tc2c_congruence_coarse,
        tc2c_congruence_fine,
    ],
    [pb_minima_coarse, pb_minima_fine, tc2c_minima_coarse, tc2c_minima_fine],
    [stac_shares_coarse, stac_shares_fine],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
