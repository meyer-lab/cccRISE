"""STACCato eigenvalue shares of each tensor mode, ALAD and IBD cohorts, fine
cell type labels."""

from ... import common
from ._panels import stac_shares

AXES = ("rect",)


def plot(axs):
    stac_shares(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
