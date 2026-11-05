"""`figures.niches.orderings` on IBD."""

from ... import common, niches

AXES = ("niches_orderings_ibd",)


def plot(axs):
    niches.orderings(axs, niches.SUPPLEMENT)


def makeFigure():
    return common.standalone_fixed(__name__)
