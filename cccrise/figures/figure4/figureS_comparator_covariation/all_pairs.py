"""Top-pair co-variation over all pairs, every component of every fit."""

from ... import common
from .._strips import covariation_spread

AXES = ("full",)


def plot(axs):
    covariation_spread(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
