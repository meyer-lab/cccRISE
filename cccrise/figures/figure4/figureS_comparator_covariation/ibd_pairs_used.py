"""How many top pairs each IBD component tests."""

from ... import common
from .._strips import pairs_used

AXES = ("full_strip_low",)


def plot(axs):
    pairs_used(axs, dataset="ibd", key=False)


def makeFigure():
    return common.standalone_fixed(__name__)
