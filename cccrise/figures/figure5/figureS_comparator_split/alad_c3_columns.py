"""ALAD C3's carrying pairs in the component and in its carrying
columns of every comparator (`changes._split`)."""

from ... import common
from .. import _split

AXES = ("split_columns_c3",)


def plot(axs):
    _split.pair_heatmap(axs[0], "alad", (3,), "ALAD C3", drop=0.16)


def makeFigure():
    return common.standalone_fixed(__name__)
