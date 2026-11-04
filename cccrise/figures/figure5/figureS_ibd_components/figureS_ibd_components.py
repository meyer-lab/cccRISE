"""Supplementary figure: the six IBD components together. Each fine cell
type's mean per-cell sender and receiver weight (as Figure 3 shows ALAD's)."""

from ... import common, style
from . import celltype_weights

LAYOUT = [[celltype_weights]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
