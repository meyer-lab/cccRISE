"""IBD: `_missed.contents`."""

from ... import common
from .._missed import contents as draw

AXES = ("missed_matrix",)


def plot(axs):
    draw(axs[0], "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
