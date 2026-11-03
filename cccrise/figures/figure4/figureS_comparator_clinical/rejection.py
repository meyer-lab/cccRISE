"""Every component's η² for acute rejection, every component of η² above 0.1 labelled."""

from ... import common
from .._strips import association_strip

AXES = ("square",)


def plot(axs):
    association_strip(axs, "Acute Rejection (Yes vs. No)", above=0.1)


def makeFigure():
    return common.standalone_fixed(__name__)
