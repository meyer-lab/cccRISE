"""Each cell's sender against receiver weight on each IBD CPD component, each
divided by its largest value, colored by sender and receiver population.

IBD: each component's per-cell sender vs. receiver weight, its
populations colored (`_niches.weight_scatters`)."""

from ... import common
from .._niches import weight_scatters

AXES = ("full_sixth",) * 6
JOINED = True  # one panel, lettered once (`common._Joined`)


def plot(axs):
    weight_scatters(axs, "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
