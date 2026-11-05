"""`figures.niches.concordance` on IBD."""

from ... import common, niches

AXES = ("niches_concordance",)


def plot(axs):
    niches.concordance(axs, "ibd", niches.IBD_SUPPLEMENT)


def makeFigure():
    return common.standalone_fixed(__name__)
