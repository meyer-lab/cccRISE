"""How many top pairs each component of every fit tests for co-variation."""

from ... import common
from .._strips import pairs_used

AXES = ("full_strip",)


def plot(axs):
    pairs_used(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
