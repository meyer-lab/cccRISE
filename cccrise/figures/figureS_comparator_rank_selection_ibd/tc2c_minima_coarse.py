"""The distinct minima of Tensor-cell2cell's random starts (coarse labels)."""

from .. import common
from ._panels import DATASET, tc2c_minima

AXES = ("square",)


def plot(axs):
    tc2c_minima(axs, DATASET, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
