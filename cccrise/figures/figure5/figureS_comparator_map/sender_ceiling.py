"""Label ceiling of each program's sender weights, with the panels' key."""

from ... import common
from .._ceiling import ceiling, key

AXES = ("map_ceiling",)


def plot(axs):
    ceiling(axs[0], "sender")
    key(axs[0], inside=True)


def makeFigure():
    return common.standalone_fixed(__name__)
