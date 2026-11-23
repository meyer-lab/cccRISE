"""Fit R2X of pseudobulk CPD for each seeded SVD start at each rank, ALAD and
IBD cohorts, fine cell type labels."""

from ... import common
from ._panels import pb_fit

AXES = ("rank_selection",)


def plot(axs):
    pb_fit(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
