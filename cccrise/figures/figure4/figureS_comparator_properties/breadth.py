"""Effective LR pairs against effective samples, every component of every fit."""

from ... import common
from .._columns import breadth

AXES = ("half",)


def plot(axs):
    breadth(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
