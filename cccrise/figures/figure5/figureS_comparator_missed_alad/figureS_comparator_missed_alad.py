"""Supplementary figure, ALAD: what the comparators find that CCC-RISE does
not. Each PB and STAC LR column's largest share of its squared loading on one
CCC-RISE component's carrying pairs (A); the strongest association of every
column and of the associated CCC-RISE components (B); and the carrying pairs
of the columns under 0.5 share, the associated ones bold and boxed (C)."""

from ... import common, style
from . import associations, contents, share

LAYOUT = [[share, associations], [contents]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
