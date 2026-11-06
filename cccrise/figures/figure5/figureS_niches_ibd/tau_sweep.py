"""`figures.niches.tau_sweep` on every IBD component."""

from ... import common, niches

AXES = ("mid",)


def plot(axs):
    niches.tau_sweep(axs, niches.ALL)


def makeFigure():
    return common.standalone_fixed(__name__)
