"""Supplementary figure: CCC-RISE's populations for NICHES (Figure 5H). Each
component's per-cell sender vs. receiver weight with its sender, receiver and
both populations, ALAD (A) and IBD (B); and each population's fraction
NICHES-positive minus what cell type predicts (within sample) across tau,
ALAD (C) and IBD (D)."""

from ... import common, style
from . import alad_tau, alad_weights, ibd_tau, ibd_weights

LAYOUT = [[alad_weights], [ibd_weights], [alad_tau, ibd_tau]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
