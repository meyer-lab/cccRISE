"""`figures.niches.tau_sweep` on IBD's components 3 and 4."""

from .. import common, niches

AXES = ("mid",)


def plot(axs):
    niches.tau_sweep(axs, niches.MAIN)


def makeFigure():
    return common.standalone_fixed(__name__)
