"""Supplementary figure: cccRISE's IBD sender and receiver populations against
NICHES: their concordance over what their labels predict, across tau, and how
weighting any direction by expression reaches NICHES (`figures.niches`)."""

from ... import niches
from . import concordance, orderings, tau_sweep


def makeFigure():
    return niches.compose([[concordance, tau_sweep], [orderings]])
