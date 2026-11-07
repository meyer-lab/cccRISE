"""Tensor-cell2cell's normalized error across random starts, on ALAD and IBD (coarse labels)."""

from ... import common
from ._panels import tc2c_elbow

AXES = ("square",)


def plot(axs):
    tc2c_elbow(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
