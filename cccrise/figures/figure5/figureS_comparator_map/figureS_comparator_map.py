"""Supplementary figure: what Figure 5A rests on. How much of each program's
sender (A) and receiver (B) weights labels can hold, and how much the closest
column does; how each comparator's coarse columns map onto its fine ones on ALAD
(C) and IBD (E); and Figure 5A on coarse labels (D)."""

from ... import common, style
from . import (
    alad_columns,
    coarse_map,
    ibd_columns,
    receiver_ceiling,
    sender_ceiling,
)

LAYOUT = [
    [sender_ceiling, receiver_ceiling, alad_columns],
    [coarse_map, ibd_columns],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
