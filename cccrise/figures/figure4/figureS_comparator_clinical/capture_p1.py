"""P1 fine's top pairs in it and its three closest columns of other fits."""

from ... import common
from .._columns import capture

AXES = ("square",)


def plot(axs):
    capture(axs, ("PB", "fine", 1), matches=3, colorbar=False, compact=True)


def makeFigure():
    return common.standalone_fixed(__name__)
