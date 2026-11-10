"""Pseudobulk ceiling of each component's receiver weights."""

from ... import common
from .._ceiling import ceiling

AXES = ("map_ceiling",)


def plot(axs):
    ceiling(axs[0], "receiver")


def makeFigure():
    return common.standalone_fixed(__name__)
