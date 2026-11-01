"""Each comparator LR column's containment in cccRISE, by audit class (fine labels)."""

from ... import common
from .._panels import outside

AXES = ("square",)


def plot(axs):
    outside(axs, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
