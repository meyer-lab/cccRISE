"""Pseudobulk CPD's fit R2X across seeded starts and its LR pair mode's collinearity per start (fine labels)."""

from ... import common
from ._panels import DATASET, pb_collinearity, pb_fit

AXES = ("square", "square")


def plot(axs):
    pb_fit(axs[:1], DATASET, "fine")
    pb_collinearity(axs[1:], DATASET, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
