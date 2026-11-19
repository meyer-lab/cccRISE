"""Schematic of a CPD component's condition, sender, receiver, and LR pair
loadings."""

from .. import common

SCHEMATIC = True  # an illustration: no source data
AXES = ("full",)


def plot(axs):
    (ax,) = axs
    common.placeholder(ax, "Component loadings: sender,\nreceiver, LR pair, condition")


def makeFigure():
    return common.standalone_fixed(__name__)
