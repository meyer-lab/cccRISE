"""Figure 4C: top-pair co-variation over gene-disjoint pairs, as 4B."""

from .. import common
from ._strips import covariation_main

AXES = ("rect",)


def plot(axs):
    covariation_main(axs, disjoint=True)


def makeFigure():
    return common.standalone_fixed(__name__)
