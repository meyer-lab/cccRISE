"""Significance concordance with Tensor-cell2cell (`significance`)."""

from functools import partial

from .. import common
from . import significance


def makeFigure():
    return common.standalone(
        partial(significance.plot, method="Tensor-cell2cell"),
        significance.FIGURE_SIZE,
    )
