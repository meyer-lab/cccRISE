"""`figures.niches.orderings` on ALAD."""

from .. import common, niches

AXES = ("niches_orderings",)


def plot(axs):
    niches.orderings(axs, "alad", niches.ALAD)


def makeFigure():
    return common.standalone_fixed(__name__)
