"""Genes differing between the sender and receiver groups
(`_volcano.plot_volcano`)."""

from ... import common
from ._volcano import plot_volcano

AXES = ("square",)


def plot(axs):
    plot_volcano(axs[0], "sender", "receiver")


def makeFigure():
    return common.standalone_fixed(__name__)
