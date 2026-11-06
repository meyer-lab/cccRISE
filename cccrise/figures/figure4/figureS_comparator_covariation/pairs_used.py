"""How many top pairs each ALAD component tests, with the key."""

from ... import common
from .._strips import pairs_used

AXES = ("full_strip_low",)


def plot(axs):
    pairs_used(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
