"""Mean LR containment between every pair of methods (coarse labels)."""

from ... import common
from .._panels import pairwise

AXES = ("square",)


def plot(axs):
    pairwise(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
