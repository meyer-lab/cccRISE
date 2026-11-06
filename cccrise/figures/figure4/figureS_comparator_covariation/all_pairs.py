"""Top-pair co-variation over all pairs, ALAD."""

from ... import common
from .._strips import covariation_spread

AXES = ("full_low",)


def plot(axs):
    covariation_spread(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
