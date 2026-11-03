"""How each comparator's LR columns shift from coarse to fine labels."""

from ... import common
from .._columns import coarse_to_fine

AXES = ("half",)


def plot(axs):
    coarse_to_fine(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
