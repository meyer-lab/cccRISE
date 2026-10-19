"""Supplementary figure: every ALAD cell's PaCMAP embedding, colored by cell
type, 6-month outcome, the outcome-associated CP component's sender and
receiver weights, and its RISE eigenstate projection."""

from .. import common, style
from . import pacmaps, weight_pacmaps

LAYOUT = [[pacmaps], [weight_pacmaps]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
