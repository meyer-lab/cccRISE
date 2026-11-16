"""Supplementary figure: the single-cell comparators. CCC-RISE's populations
for NICHES (Figure 5J): each component's per-cell sender vs. receiver weight
with its sender, receiver and both populations, ALAD (A) and IBD (B); each
population's fraction NICHES-positive minus what cell type predicts (within
sample) across tau, ALAD (C) and IBD (D). Why Scriabin is not one: its
interaction programs do not reproduce between seeds, CCC-RISE's components'
carrying LR pairs do (E)."""

from ... import common, style
from . import alad_tau, alad_weights, ibd_tau, ibd_weights, scriabin_seeds

# Both cohorts have six components, a full row each; Scriabin sits beside the tau panels
LAYOUT = [[alad_weights], [ibd_weights], [alad_tau, ibd_tau, scriabin_seeds]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
