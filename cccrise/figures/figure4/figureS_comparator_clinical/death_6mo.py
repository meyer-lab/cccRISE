"""Every component's η² for 6-month death."""

from ... import common
from .._strips import association_strip

AXES = ("square",)


def plot(axs):
    association_strip(axs, "6-Month Died vs. Alive")


def makeFigure():
    return common.standalone_fixed(__name__)
