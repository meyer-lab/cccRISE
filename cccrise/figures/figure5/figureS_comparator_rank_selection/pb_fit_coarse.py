"""Fit R2X of pseudobulk CPD for each seeded SVD start at each rank, ALAD and
IBD cohorts, coarse cell type labels."""

from ... import common
from ._panels import pb_fit

AXES = ("rank_selection",)


def plot(axs):
    pb_fit(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
