"""Placeholder for an illustration of a CP component's sender, receiver, LR
pair, and condition loadings."""

from .. import common

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    common.placeholder(ax, "Component loadings: sender,\nreceiver, LR pair, condition")


def makeFigure():
    return common.standalone_fixed(__name__)
