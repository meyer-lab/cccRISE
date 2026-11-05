"""Figure 4J: the label ceiling of each program's receiver weight."""

from .. import common
from ._ceiling import ceiling, key

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    ceiling(ax, "receiver")
    key(ax)


def makeFigure():
    return common.standalone_fixed(__name__)
