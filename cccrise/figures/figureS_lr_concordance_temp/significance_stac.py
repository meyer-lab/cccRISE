"""Significance concordance with STACCato (`significance`)."""

from functools import partial

from .. import common, style
from . import significance


def makeFigure():
    return common.standalone(
        partial(significance.plot, method="STACCato"), (125 * style.MM, 85 * style.MM)
    )
