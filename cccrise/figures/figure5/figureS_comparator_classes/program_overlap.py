"""Overlap of every column not contained with one cccRISE program."""

from ... import common
from .._classes import overlap

AXES = ("half",)


def plot(axs):
    overlap(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
