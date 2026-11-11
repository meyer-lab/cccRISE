"""Supplementary figure, IBD: what the comparators find that CCC-RISE does
not. Each PB and STAC LR column's containment in CCC-RISE's LR span (A); the
strongest association of every column and of the associated CCC-RISE
components (B); and the carrying pairs of the columns under 0.5
containment, the associated ones bold and boxed (C)."""

from ... import common, style
from . import associations, containment, contents

LAYOUT = [[containment, associations], [contents]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
