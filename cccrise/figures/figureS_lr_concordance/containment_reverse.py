"""Each comparator column's LR containment in cccRISE's span
(`containment.plot_reverse`)."""

from .. import common, style
from . import containment


def makeFigure():
    return common.standalone(
        containment.plot_reverse, (style.FIGURE_WIDTH_DOUBLE / 2, 110 * style.MM)
    )
