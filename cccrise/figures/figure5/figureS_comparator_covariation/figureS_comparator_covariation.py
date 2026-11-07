"""Supplementary figure: the components above the stricter null in either
version, every fit, ALAD over IBD, aligned down each cohort's panels: how many
top pairs each tests (A, D), and their co-variation over all pairs of pairs (B,
E) and over only those sharing no gene (C, F)."""

from ... import common, style
from . import (
    all_pairs,
    disjoint,
    ibd_all_pairs,
    ibd_disjoint,
    ibd_pairs_used,
    pairs_used,
)

LAYOUT = [
    [[pairs_used, all_pairs, disjoint, ibd_pairs_used, ibd_all_pairs, ibd_disjoint]]
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
