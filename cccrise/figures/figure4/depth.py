"""Figure 4D: how much each component's sample profile tracks depth."""

from .. import common
from ._strips import depth

AXES = ("mid",)


def plot(axs):
    depth(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
