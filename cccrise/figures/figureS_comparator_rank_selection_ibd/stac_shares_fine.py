"""STACCato's eigenvalue shares per mode (fine labels)."""

from .. import common
from ._panels import DATASET, stac_shares

AXES = ("square",)


def plot(axs):
    stac_shares(axs, DATASET, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
