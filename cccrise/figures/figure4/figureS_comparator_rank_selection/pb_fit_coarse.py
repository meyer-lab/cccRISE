"""Pseudobulk CPD's fit R2X across seeded starts, on ALAD and IBD (coarse labels)."""

from ... import common
from ._panels import pb_fit

AXES = ("square",)


def plot(axs):
    pb_fit(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
