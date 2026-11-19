"""Share of pseudobulk CPD's seeded SVD starts reaching its most common minimum
at each rank, ALAD and IBD cohorts, coarse cell type labels."""

from ... import common
from ._panels import pb_minima

AXES = ("square",)


def plot(axs):
    pb_minima(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
