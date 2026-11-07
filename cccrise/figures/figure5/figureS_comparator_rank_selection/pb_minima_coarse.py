"""The share of pseudobulk CPD's starts at its most common minimum, on ALAD and IBD (coarse labels)."""

from ... import common
from ._panels import pb_minima

AXES = ("square",)


def plot(axs):
    pb_minima(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
