"""The distinct minima of Tensor-cell2cell's random starts (fine labels)."""

from ... import common
from ._panels import DATASET, tc2c_minima

AXES = ("square",)


def plot(axs):
    tc2c_minima(axs, DATASET, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
