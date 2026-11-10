"""Co-variation of IBD C1 and C5 and of their related columns, over all carrying
pairs against the random and the expression-matched null."""

from ... import common
from .. import _split

AXES = ("split_covariation_ibd",)


def plot(axs):
    _split.covariation(axs, "ibd", (1, 5))


def makeFigure():
    return common.standalone_fixed(__name__)
