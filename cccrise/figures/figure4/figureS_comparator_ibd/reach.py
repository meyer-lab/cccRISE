"""Figure 4A on IBD: cccRISE's programs' LR containment in each comparator
fit."""

from ... import common
from .._columns import reach

AXES = ("square",)


def plot(axs):
    reach(axs, "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
