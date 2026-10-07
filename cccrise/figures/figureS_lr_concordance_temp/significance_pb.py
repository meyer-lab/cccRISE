"""Significance concordance with Pseudobulk CPD (`significance`)."""

from functools import partial

from .. import common, style
from . import significance


def makeFigure():
    return common.standalone(
        partial(significance.plot, method="Pseudobulk CPD"),
        (125 * style.MM, 85 * style.MM),
    )
