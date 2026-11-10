"""Supplementary figure: what Figure 5A rests on. How much of each component's
sender (A) and receiver (B) weights pseudobulk can hold, and how much the
closest comparator column does, each on the best reading across cell-type
fits; and how far every method's components track detection depth (C)."""

from ... import common, style
from . import depth_tracking, receiver_ceiling, sender_ceiling

LAYOUT = [[sender_ceiling, receiver_ceiling], [depth_tracking]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
