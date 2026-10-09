"""Draft panel D, coarse-annotation comparator fits only (`missed`)."""

from functools import partial

from .. import common, style
from . import missed


def makeFigure():
    return common.standalone(
        partial(missed.plot, annotations=("coarse",)),
        (style.FIGURE_WIDTH_DOUBLE, 80 * style.MM),
    )
