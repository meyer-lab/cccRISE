"""`figures.niches.concordance` on ALAD."""

from .. import common, niches

AXES = ("niches_concordance",)


def plot(axs):
    niches.concordance(axs, "alad", niches.ALAD, legend_stack=0)


def makeFigure():
    return common.standalone_fixed(__name__)
