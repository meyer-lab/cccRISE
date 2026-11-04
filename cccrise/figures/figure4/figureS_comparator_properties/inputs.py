"""Each method's input tensor against sequencing depth."""

from ... import common
from .._columns import inputs

AXES = ("half",)


def plot(axs):
    inputs(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
