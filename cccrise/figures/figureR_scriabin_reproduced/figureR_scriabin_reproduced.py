"""Reviewer figure: the Scriabin paper's robustness analyses (Wilk et al.
2024) rerun on its panc8 data with its stated settings. A-B: Fig. 2e's
technical-noise analysis on the Fluidigm C1 islets (downsampled to inDrop
coverage; LISI of the ground-truth cell-cell pairs after integration).
D-K: Extended Data Fig. 8f's soft-power sweep on random inDrop subsamples
of 300 (D-G) and 1,000 cells (H-K). C, L-M: the read-out the paper does not
make, whether the same interaction programs come back, on the same data.
Grey: the paper's values."""

from .. import common, style
from . import (
    downsampling_fold,
    downsampling_programs,
    lisi_recapitulation,
    soft_power_300,
    soft_power_1000,
    soft_power_programs,
)

LAYOUT = [
    [lisi_recapitulation, downsampling_fold, downsampling_programs],
    [soft_power_300],
    [soft_power_1000],
    [soft_power_programs],
]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
