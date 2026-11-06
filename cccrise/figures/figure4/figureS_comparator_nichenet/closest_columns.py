"""Each program's score against its closest columns."""

from ... import common
from .._nichenet import closest_strip

AXES = ("mid",)


def plot(axs):
    closest_strip(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
