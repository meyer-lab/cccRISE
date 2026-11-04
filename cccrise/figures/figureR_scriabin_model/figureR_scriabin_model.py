"""Reviewer figure: Scriabin's interaction programs do not reproduce between
random seeds at the published defaults, whatever the dataset or
preprocessing, while cccRISE's components do (A); agreement breaks where
modules are partitioned out of a reproducible network (B); and only the
full cell-cell interaction matrix together with tuned settings makes the
programs reproducible both between seeds and when one sample or cell type
is left out (C)."""

from .. import common, style
from . import remedy, seed_agreement, stage_agreement

LAYOUT = [[seed_agreement, stage_agreement, remedy]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
