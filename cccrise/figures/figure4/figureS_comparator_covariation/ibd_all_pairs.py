"""Top-pair co-variation over all pairs, IBD."""

from ... import common
from .._strips import covariation_spread

AXES = ("full_low",)


def plot(axs):
    covariation_spread(axs, dataset="ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
