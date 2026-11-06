"""The share of Tensor-cell2cell's starts at its most common minimum, on ALAD and IBD (coarse labels)."""

from ... import common
from ._panels import tc2c_minima

AXES = ("square",)


def plot(axs):
    tc2c_minima(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
