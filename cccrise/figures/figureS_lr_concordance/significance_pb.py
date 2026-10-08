"""Significance concordance with Pseudobulk CPD (`significance`)."""

from functools import partial

from .. import common
from . import significance


def makeFigure():
    return common.standalone(
        partial(significance.plot, method="Pseudobulk CPD"),
        significance.FIGURE_SIZE,
    )
