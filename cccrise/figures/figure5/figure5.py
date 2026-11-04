"""Figure 5: cccRISE on the IBDverse terminal-ileum cohort.

The CP factor matrices (A-D); the EE/EC/tuft map, the hormone programs'
sender weights, and their ligands GCG and PYY on its EE/EC part (E-H); the hormone
programs resolve two functional hormone states inside the annotated
enteroendocrine type, villus-like (C3) and crypt-like (C4), and inflammation
depletes the C3-only villus-like cells, not the C3-high ones (I-J), while
the C4-high share of the tuft receivers rises (K)."""

from .. import common, style
from . import (
    cp_factor_matrices,
    ee_state_map,
    hormone_expression_map,
    hormone_receiver_share,
    hormone_state_markers,
    hormone_state_share,
)

LAYOUT = [
    [cp_factor_matrices],
    [ee_state_map, hormone_expression_map],
    [hormone_state_markers, hormone_state_share, hormone_receiver_share],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
