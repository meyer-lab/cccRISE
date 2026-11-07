"""Effective LR pairs of every comparator column."""

from ... import common
from .._classes import breadth

AXES = ("half",)


def plot(axs):
    breadth(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
