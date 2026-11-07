"""The share of pseudobulk CPD's starts at its most common minimum, on ALAD and IBD (fine labels)."""

from ... import common
from ._panels import pb_minima

AXES = ("square",)


def plot(axs):
    pb_minima(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
