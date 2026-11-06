"""Tensor-cell2cell's largest congruence product per start, on ALAD and IBD (coarse labels)."""

from ... import common
from ._panels import tc2c_congruence

AXES = ("square",)


def plot(axs):
    tc2c_congruence(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
