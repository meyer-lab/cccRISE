"""How each comparator's coarse LR columns map onto its fine ones on ALAD."""

from ... import common
from .._coarse_fine import coarse_to_fine, match_label

AXES = ("map_columns",)


def plot(axs):
    coarse_to_fine(axs[0], "alad", "ALAD")
    match_label(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
