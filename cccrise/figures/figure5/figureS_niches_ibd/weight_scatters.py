"""`figures.niches.weight_scatters` on every IBD component."""

from ... import common, niches

N_COMPONENTS = 6
AXES = ("niches_scatter",) * N_COMPONENTS
ONE_LETTER = True


def plot(axs):
    niches.weight_scatters(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
