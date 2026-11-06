"""How each comparator's coarse LR columns map onto its fine ones on ALAD."""

from ... import common
from .._coarse_fine import coarse_to_fine

AXES = ("map_columns",)


def plot(axs):
    coarse_to_fine(axs[0], "alad", "ALAD")


def makeFigure():
    return common.standalone_fixed(__name__)
