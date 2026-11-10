"""ALAD: `_missed.containment`."""

from ... import common
from .._missed import containment as draw

AXES = ("missed_bars",)


def plot(axs):
    draw(axs[0], "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
