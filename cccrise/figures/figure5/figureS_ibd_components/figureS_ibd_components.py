"""Supplementary figure: the six IBD components together. Each fine cell
type's mean per-cell sender and receiver weight (as Figure 3 shows ALAD's);
every component's inflammation-status effect alone and adjusted for
composition or dissociation; and how many biopsies carry each component."""

from ... import common, style
from . import celltype_weights, loading_concentration, status_adjustment

LAYOUT = [[celltype_weights], [status_adjustment, loading_concentration]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
