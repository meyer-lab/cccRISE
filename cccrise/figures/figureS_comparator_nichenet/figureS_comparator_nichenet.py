"""Supplementary figure: NicheNet downstream-target validation of cccRISE and
the cell-type pseudobulk comparators, panel by panel as in Figure 2
(`figure2.nichenet`), and every method compared on one scale.

Comparator receivers are cell types: each cell is given its type's receiver
loading, with ties at the high/low cut broken at random
(`validation.nichenet_comparators`). STACCato (Tucker) has no per-component
receiver loading and is not scored."""

from functools import partial
from types import SimpleNamespace

from .. import common, style
from ..figure2.nichenet import plot_scores
from . import comparison
from ._common import FITS, all_scores, fit_label


def _plot_fit(method, rank_mode, fig):
    scores = all_scores()
    fit = scores[(scores["method"] == method) & (scores["rank_mode"] == rank_mode)]
    plot_scores(fig, fit, title=fit_label(method, rank_mode, scores))


PANELS = {
    f"{method}|{mode}": SimpleNamespace(plot=partial(_plot_fit, method, mode))
    for method, mode in FITS
} | {"comparison": comparison}
LAYOUT = [[f"{m}|{r}"] for m, r in FITS] + [["comparison"]]


def makeFigure():
    height = (len(FITS) + 1) * 55 * style.MM
    return common.compose((style.FIGURE_WIDTH_DOUBLE, height), LAYOUT, PANELS)
