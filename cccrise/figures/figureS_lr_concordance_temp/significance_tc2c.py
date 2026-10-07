"""Significance concordance with Tensor-cell2cell (`significance`)."""

from functools import partial

from .. import common, style
from . import significance


def makeFigure():
    return common.standalone(
        partial(significance.plot, method="Tensor-cell2cell"),
        (125 * style.MM, 85 * style.MM),
    )
