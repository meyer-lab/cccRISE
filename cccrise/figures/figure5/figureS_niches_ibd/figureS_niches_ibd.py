"""Supplementary figure: cccRISE's IBD sender and receiver populations against
NICHES: their concordance over what their labels predict, across tau, and how
weighting any direction by expression reaches NICHES (`figures.niches`)."""

from ... import common, style
from . import concordance, orderings, tau_sweep, weight_scatters


def makeFigure():
    return common.compose_fixed(
        [[weight_scatters], [concordance, tau_sweep], [orderings]],
        width=style.FIGURE_WIDTH_DOUBLE,
    )
