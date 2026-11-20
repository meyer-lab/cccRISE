"""Share of Tensor-cell2cell's random starts reaching its most common minimum
at each rank, ALAD and IBD cohorts, coarse cell type labels."""

from ... import common
from ._panels import tc2c_minima

AXES = ("square",)


def plot(axs):
    tc2c_minima(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
