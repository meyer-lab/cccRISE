"""Supplementary figure: IBD's comparator recovery grids, as Figure 5A-B show
ALAD's, and depth tracking in both cohorts. Each IBD component's best single
comparator LR column (A) and its containment in each fit's LR span (B); how
far every method's components track detection depth, ALAD beside IBD (C)."""

from ... import common, style
from . import depth, ibd_best, ibd_containment

LAYOUT = [[ibd_best, ibd_containment], [depth]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
