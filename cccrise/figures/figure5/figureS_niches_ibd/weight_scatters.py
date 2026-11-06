"""`figures.niches.weight_scatters` on every IBD component."""

from ... import common, niches

N_COMPONENTS = 6
AXES = ("full_sixth",) * N_COMPONENTS
JOINED = True  # one panel, lettered once (`common._Joined`)


def plot(axs):
    niches.weight_scatters(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
