"""Top-pair co-variation over gene-disjoint pairs, IBD."""

from ... import common
from .._strips import covariation_spread

AXES = ("full_low",)


def plot(axs):
    covariation_spread(axs, disjoint=True, dataset="ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
