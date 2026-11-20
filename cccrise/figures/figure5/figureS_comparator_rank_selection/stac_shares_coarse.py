"""STACCato eigenvalue shares of each tensor mode, ALAD and IBD cohorts, coarse
cell type labels."""

from ... import common
from ._panels import stac_shares

AXES = ("rect",)


def plot(axs):
    stac_shares(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
