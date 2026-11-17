"""Reviewer figure: the Scriabin paper's technical-noise analysis (Wilk et al.
2024, Fig. 2d-e) rerun with upstream Scriabin on its Fluidigm C1 pancreas
islets (panc8), downsampled to inDrop depth. A: the paper's read-out, LISI of
the ground-truth cell-cell pairs after integrating the downsampled CCIM with
the full-depth one (grey: the paper's values). B: the read-out the paper does
not make, whether the same interaction programs come back on the same cells."""

from .. import common
from . import downsampling_programs, lisi_recapitulation

LAYOUT = [[lisi_recapitulation, downsampling_programs]]


def makeFigure():
    return common.compose_fixed(LAYOUT)
