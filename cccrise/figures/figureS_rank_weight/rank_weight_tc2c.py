"""Loading by rank for Tensor-cell2cell's columns (`rank_weight`)."""

from functools import partial

from .. import common, style
from . import rank_weight


def makeFigure():
    return common.standalone(
        partial(rank_weight.plot, method="Tensor-cell2cell"),
        (style.FIGURE_WIDTH_DOUBLE, 65 * style.MM),
    )
