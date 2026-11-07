"""Pseudobulk CPD's LR pair mode's collinearity per start, on ALAD and IBD (fine labels)."""

from ... import common
from ._panels import pb_collinearity

AXES = ("square",)


def plot(axs):
    pb_collinearity(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
