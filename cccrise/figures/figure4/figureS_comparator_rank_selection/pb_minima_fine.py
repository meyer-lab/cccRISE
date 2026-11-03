"""The distinct minima of pseudobulk CPD's seeded starts (fine labels)."""

from ... import common
from ._panels import DATASET, pb_minima

AXES = ("square",)


def plot(axs):
    pb_minima(axs, DATASET, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
