"""Each cccRISE program's sample columns in each comparator (coarse labels)."""

from ... import common
from .._panels import reach

AXES = ("square",)


def plot(axs):
    reach(axs, "coarse", mode="sample")


def makeFigure():
    return common.standalone_fixed(__name__)
