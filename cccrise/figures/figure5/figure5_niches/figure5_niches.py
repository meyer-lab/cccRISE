"""Figure 5 NICHES row: cccRISE's IBD component 3 and 4 sender and receiver
populations against NICHES: their concordance over what their labels predict
(A) above how weighting any direction by expression reaches NICHES (C), beside
their concordance across tau (B) (`figures.niches`)."""

from ... import niches
from . import concordance, orderings, tau_sweep


def makeFigure():
    return niches.compose([[[concordance, orderings], tau_sweep]])
