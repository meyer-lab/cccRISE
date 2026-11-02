"""Pseudobulk CPD's fit R2X across seeded starts and its LR pair mode's collinearity per start (coarse labels)."""

from ... import common
from ._panels import DATASET, pb_collinearity, pb_fit

AXES = ("square", "square")


def plot(axs):
    pb_fit(axs[:1], DATASET, "coarse")
    pb_collinearity(axs[1:], DATASET, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
