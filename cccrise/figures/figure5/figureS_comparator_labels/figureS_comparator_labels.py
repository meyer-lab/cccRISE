"""Supplementary figure: what each method keeps and loses when its cells are
described more finely. How each of PB's and STAC's coarse LR columns maps onto
its fine ones, and CCC-RISE's LR columns at a reduced RISE rank onto the
reported fit's, by LR cos^2 on CCC-RISE's LR pairs, on ALAD (A) and IBD (B);
associated coarse columns that keep no fine column boxed (`_coarse_fine`)."""

from ... import common, style
from . import alad_columns, ibd_columns

LAYOUT = [[alad_columns, ibd_columns]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
