"""`figures.niches.orderings` on IBD."""

from .. import common, niches

AXES = ("niches_orderings",)


def plot(axs):
    niches.orderings(axs, "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
