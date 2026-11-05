"""Figure 4I: the label ceiling of each program's sender weight."""

from .. import common
from ._ceiling import ceiling

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    ceiling(ax, "sender")


def makeFigure():
    return common.standalone_fixed(__name__)
