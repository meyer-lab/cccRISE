"""Normalized error of Tensor-cell2cell for each random start at each rank,
ALAD and IBD cohorts, coarse cell type labels."""

from ... import common
from ._panels import tc2c_elbow

AXES = ("rank_selection",)


def plot(axs):
    tc2c_elbow(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
