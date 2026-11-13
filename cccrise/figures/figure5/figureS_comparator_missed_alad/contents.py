"""ALAD: `_missed.contents`."""

from ... import common
from .._missed import contents as draw

AXES = ("missed_matrix",)


def plot(axs):
    draw(axs[0], "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
