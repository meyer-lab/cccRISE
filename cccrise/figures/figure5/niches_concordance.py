"""`figures.niches.concordance` on IBD's components 3 and 4."""

from .. import common, niches

AXES = ("rect",)


def plot(axs):
    niches.concordance(axs, niches.MAIN, legend="below")


def makeFigure():
    return common.standalone_fixed(__name__)
