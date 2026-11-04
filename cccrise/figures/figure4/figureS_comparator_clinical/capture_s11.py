"""S11 coarse's top pairs in it and its three closest columns of other fits."""

from ... import common
from .._columns import capture

AXES = ("square",)


def plot(axs):
    capture(axs, ("STAC", "coarse", 11), matches=3, colorbar=True, compact=True)


def makeFigure():
    return common.standalone_fixed(__name__)
