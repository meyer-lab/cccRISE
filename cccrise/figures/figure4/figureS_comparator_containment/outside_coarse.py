"""Each comparator LR column's containment in cccRISE, by audit class (coarse labels)."""

from ... import common
from .._panels import outside

AXES = ("square",)


def plot(axs):
    outside(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
