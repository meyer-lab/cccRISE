"""`figures.niches.tau_sweep` on IBD's components 3 and 4."""

from ... import common, niches

AXES = ("niches_tau_main",)


def plot(axs):
    niches.tau_sweep(axs, "ibd", niches.IBD_MAIN)


def makeFigure():
    return common.standalone_fixed(__name__)
