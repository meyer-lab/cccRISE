"""Figure 5: cccRISE on the IBDverse terminal-ileum cohort.

The CP factor matrices (A-D); the EE/EC/tuft map, the hormone programs'
sender weights and their ligands GCG and PYY on it (E-H); the hormone
programs resolve two functional hormone states inside the annotated
enteroendocrine type, villus-like (C3) and crypt-like (C4), and inflammation
depletes the villus-like one (I-K);
the inflammatory programs (C1, C5) are one coordinated circuit, their top
pairs co-varying across biopsies (Figure 3D's test) and their receivers
expressing the ligands' downstream targets (NicheNet, as Figure 2) (L-M)."""

from .. import common, style
from . import (
    cp_factor_matrices,
    ee_state_map,
    hormone_contour_map,
    hormone_state_markers,
    hormone_state_share,
    l_hormone_weights,
    nichenet,
    program_covariation,
)

LAYOUT = [
    [cp_factor_matrices],
    [ee_state_map, hormone_contour_map],
    [l_hormone_weights, hormone_state_markers, hormone_state_share],
    [program_covariation, nichenet],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
