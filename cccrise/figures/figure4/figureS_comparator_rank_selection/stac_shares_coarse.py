"""STACCato's eigenvalue shares per mode (coarse labels)."""

from ... import common
from ._panels import DATASET, stac_shares

AXES = ("square",)


def plot(axs):
    stac_shares(axs, DATASET, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
