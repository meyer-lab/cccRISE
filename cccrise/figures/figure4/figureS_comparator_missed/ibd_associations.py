"""The IBD left-over columns' associations."""

from ... import common
from .._missed import associations

AXES = ("missed_associations",)


def plot(axs):
    associations(axs[0], "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
