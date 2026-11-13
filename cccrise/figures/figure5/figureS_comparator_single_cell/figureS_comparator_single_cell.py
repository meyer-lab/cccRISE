"""Supplementary figure: the single-cell comparators. Why Scriabin is not one:
its interaction programs do not reproduce between seeds, CCC-RISE's
components' carrying LR pairs do (A). CCC-RISE's populations for NICHES
(Figure 5J): each component's per-cell sender vs. receiver weight with its
sender, receiver and both populations, ALAD (B) and IBD (C); and each
population's fraction NICHES-positive minus what cell type predicts (within
sample) across tau, ALAD (D) and IBD (E)."""

from ... import common, style
from . import alad_tau, alad_weights, ibd_tau, ibd_weights, scriabin_seeds

# Scriabin takes the sixth ALAD's five components leave free
LAYOUT = [[scriabin_seeds, alad_weights], [ibd_weights], [alad_tau, ibd_tau]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
