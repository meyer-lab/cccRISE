"""ALAD: each component's per-cell sender vs. receiver weight, its
populations colored (`_niches.weight_scatters`)."""

from ... import common
from .._niches import weight_scatters

AXES = ("full_sixth",) * 5
JOINED = True  # one panel, lettered once (`common._Joined`)


def plot(axs):
    weight_scatters(axs, "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
