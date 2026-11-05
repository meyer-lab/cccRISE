"""`figures.niches.tau_sweep` on IBD."""

from .. import common, niches

AXES = ("mid",)


def plot(axs):
    niches.tau_sweep(axs, "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
