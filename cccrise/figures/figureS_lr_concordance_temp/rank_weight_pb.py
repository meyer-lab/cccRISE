"""Loading by rank for Pseudobulk CPD's columns (`rank_weight`)."""

from functools import partial

from .. import common, style
from . import rank_weight


def makeFigure():
    return common.standalone(
        partial(rank_weight.plot, method="Pseudobulk CPD"),
        (style.FIGURE_WIDTH_DOUBLE, 65 * style.MM),
    )
