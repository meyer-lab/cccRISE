"""`figures.niches.concordance` on IBD's components 3 and 4."""

from ... import common, niches

AXES = ("niches_concordance_main",)


def plot(axs):
    niches.concordance(axs, "ibd", niches.IBD_MAIN, legend_stack=None)


def makeFigure():
    return common.standalone_fixed(__name__)
