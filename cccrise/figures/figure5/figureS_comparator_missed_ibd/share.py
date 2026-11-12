"""IBD: `_missed.share`."""

from ... import common
from .._missed import share as draw

AXES = ("missed_bars",)


def plot(axs):
    draw(axs[0], "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
