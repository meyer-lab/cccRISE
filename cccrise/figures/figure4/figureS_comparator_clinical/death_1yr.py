"""Every component's η² for 1-year death, every component of η² above 0.1 labelled."""

from ... import common
from .._strips import association_strip

AXES = ("square",)


def plot(axs):
    association_strip(axs, "1-Year Died vs. Alive", above=0.1)


def makeFigure():
    return common.standalone_fixed(__name__)
