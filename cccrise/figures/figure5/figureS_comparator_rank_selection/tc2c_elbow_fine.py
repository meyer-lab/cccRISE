"""Normalized error of Tensor-cell2cell for each random start at each rank,
ALAD and IBD cohorts, fine cell type labels."""

from ... import common
from ._panels import tc2c_elbow

AXES = ("square",)


def plot(axs):
    tc2c_elbow(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
