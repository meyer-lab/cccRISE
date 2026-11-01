"""Figure 4H: cccRISE C4's top pairs in C4 and its closest comparator columns."""

from .. import common
from ._columns import capture

AXES = ("rect",)


def plot(axs):
    capture(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
