"""Figure 4F: what each LR column is."""

from .. import common
from ._strips import classes

AXES = ("mid",)


def plot(axs):
    classes(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
