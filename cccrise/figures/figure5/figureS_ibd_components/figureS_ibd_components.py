"""Supplementary figure: the six IBD components together. Each fine cell
type's mean per-cell sender and receiver weight (as Figure 3 shows ALAD's),
and how many biopsies carry each component."""

from ... import common, style
from . import celltype_weights, loading_concentration

LAYOUT = [[celltype_weights], [loading_concentration]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
