"""Each cccRISE program's sample columns in each comparator (fine labels)."""

from ... import common
from .._panels import reach

AXES = ("square",)


def plot(axs):
    reach(axs, "fine", mode="sample")


def makeFigure():
    return common.standalone_fixed(__name__)
