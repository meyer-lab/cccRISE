"""How many top pairs each IBD component tests, with the key above."""

from ... import common
from .._strips import pairs_used

AXES = ("full_strip_low",)


def plot(axs):
    pairs_used(axs, dataset="ibd", key=True)


def makeFigure():
    return common.standalone_fixed(__name__)
