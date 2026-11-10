"""IBD C1 and C5 together: their carrying pairs in each component and in
their related columns of each comparator, with each comparator's best
combination (Σ); a column carrying both lists both shares, C1 first."""

from ... import common
from .. import _split

AXES = ("split_columns_ibd",)


def plot(axs):
    _split.pair_heatmap(axs[0], "ibd", (1, 5), "IBD C1 and C5", drop=0.3, signed=True)


def makeFigure():
    return common.standalone_fixed(__name__)
