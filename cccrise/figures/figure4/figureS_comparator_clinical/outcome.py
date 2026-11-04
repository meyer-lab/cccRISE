"""Every component's η² for 6-month outcome."""

from ... import common
from .._strips import association_strip

AXES = ("square",)


def plot(axs):
    association_strip(axs, "6-Month Outcome (4-Way)")


def makeFigure():
    return common.standalone_fixed(__name__)
