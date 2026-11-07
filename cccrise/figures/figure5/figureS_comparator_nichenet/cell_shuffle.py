"""Weighted target score of every column above shuffled cells."""

from ... import common
from .._nichenet import cell_shuffle_strip

AXES = ("mid",)


def plot(axs):
    cell_shuffle_strip(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
