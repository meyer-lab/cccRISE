"""Supplementary figure: NicheNet validation against a shuffled prior
instead of Figure 2's shuffled cells: the ligands' weight-counted target-score
percentiles, and their test against the shuffled prior."""

from ... import common, style
from . import prior_shuffle, weighted_percentiles

LAYOUT = [[weighted_percentiles, prior_shuffle]]


def makeFigure():
    return common.compose_fixed(LAYOUT, width=style.FIGURE_WIDTH_DOUBLE)
