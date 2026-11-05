"""Supplementary figure: Figure 4A and E on IBD. Which of cccRISE's IBD
programs each comparator fit contains (A), and which comparator columns
cccRISE's span contains (B). Each comparator is fit on IBD's lineages (coarse)
and fine types."""

from ... import common, style
from . import columns_in_cccrise, reach

LAYOUT = [[reach, columns_in_cccrise]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
