"""Figure 5D: co-variation of ALAD C5 and of its related columns (5C) over all
their carrying pairs, against the random and the expression-matched null
(`_split.covariation`)."""

from .. import common
from . import _split

AXES = ("square",)


def plot(axs):
    _split.covariation(axs, "alad", (5,), key_loc="upper right")


def makeFigure():
    return common.standalone_fixed(__name__)
