"""`figures.niches.tau_sweep` on ALAD."""

from .. import common, niches

AXES = ("mid",)


def plot(axs):
    niches.tau_sweep(axs, "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
