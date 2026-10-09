"""Draft panel D, comparator fits on the coarse and fine annotations (`missed`)."""

from functools import partial

from .. import common, style
from . import missed


def makeFigure():
    return common.standalone(
        partial(missed.plot, annotations=("coarse", "fine")),
        (style.FIGURE_WIDTH_DOUBLE, 80 * style.MM),
    )
