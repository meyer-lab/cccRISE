"""Extended Data Fig. 8f rerun on random subsamples of 300 inDrop cells
(`_sweep`)."""

from .. import common
from . import _sweep

AXES = _sweep.AXES


def plot(axs):
    _sweep.draw(axs, cells=300)


def makeFigure():
    return common.standalone_fixed(__name__)
