"""Supplementary figure: how much weight an LR column keeps down its ranking,
for cccRISE (A) and each comparator's coarse own-rank fit (B-D); left, each
column's |loading| relative to its largest; right, the cumulative share of
its squared loading."""

from functools import partial
from types import SimpleNamespace

from .. import common, style
from ..figureS_lr_concordance._common import METHODS
from . import rank_weight


def _plot(fig, method):
    # One level down, so the method's title sits under the panel letter.
    rank_weight.plot(fig.subfigures(1, 1), method)


PANELS = {
    method: SimpleNamespace(plot=partial(_plot, method=method))
    for method in ("cccRISE", *METHODS)
}
LAYOUT = [[method] for method in PANELS]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX), LAYOUT, PANELS
    )
