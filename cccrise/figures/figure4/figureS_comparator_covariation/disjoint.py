"""Top-pair co-variation over gene-disjoint pairs, every component of every fit."""

from ... import common
from .._strips import covariation_spread

AXES = ("full",)


def plot(axs):
    covariation_spread(axs, disjoint=True)


def makeFigure():
    return common.standalone_fixed(__name__)
