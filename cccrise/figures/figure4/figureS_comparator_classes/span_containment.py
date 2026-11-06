"""Containment of every comparator column in cccRISE's span."""

from ... import common
from .._classes import containment

AXES = ("half",)


def plot(axs):
    containment(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
