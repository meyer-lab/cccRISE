"""IBD's mixed programs' pairs in STAC's columns."""

from ... import common
from .._recovery import pair_columns

AXES = ("split_pairs",)


def plot(axs):
    pair_columns(axs[0], "ibd", (1, 5), title=None, outline=False)


def makeFigure():
    return common.standalone_fixed(__name__)
