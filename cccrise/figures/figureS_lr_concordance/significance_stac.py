"""Significance concordance with STACCato (`significance`)."""

from functools import partial

from .. import common
from . import significance


def makeFigure():
    return common.standalone(
        partial(significance.plot, method="STACCato"), significance.FIGURE_SIZE
    )
