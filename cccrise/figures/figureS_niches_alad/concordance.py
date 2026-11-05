"""`figures.niches.concordance` on ALAD."""

from .. import common, niches

AXES = ("niches_concordance",)


def plot(axs):
    niches.concordance(axs, "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
