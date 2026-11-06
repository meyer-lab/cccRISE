"""`figures.niches.orderings` on IBD's components 3 and 4."""

from .. import common, niches

AXES = ("wide",)


def plot(axs):
    niches.orderings(axs, niches.MAIN, labels=False)


def makeFigure():
    return common.standalone_fixed(__name__)
