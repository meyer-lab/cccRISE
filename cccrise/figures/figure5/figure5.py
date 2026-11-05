"""Figure 5: cccRISE on the IBDverse terminal-ileum cohort.

The CP factor matrices (A-D); the EE/EC/tuft map, the hormone programs'
sender weights, and their ligands PYY and GCG on its EE/EC part (E-H); each
EE cell's C3/C4 sender-weight balance as an angle (I-J), whose extremes
resolve two functional hormone states inside the annotated enteroendocrine
type, villus-like (C3) and crypt-like (C4) (K); inflammation depletes the
villus-like cells (S_ibd_hormone_axes)."""

from .. import common, style
from . import (
    cp_factor_matrices,
    ee_state_map,
    hormone_angle_hist,
    hormone_angle_scatter,
    hormone_expression_map,
    hormone_extreme_markers,
)

LAYOUT = [
    [cp_factor_matrices],
    [ee_state_map, hormone_expression_map],
    [hormone_angle_scatter, hormone_angle_hist, hormone_extreme_markers],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
