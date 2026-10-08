"""Supplementary figure: how much weight an LR column keeps down its ranking.
cccRISE (A, top centre), then each comparator's own-rank fits, one row per
comparator and one column per annotation (coarse, fine). In each panel, left:
each column's |loading| relative to its largest; right: the cumulative share
of its squared loading."""

from functools import partial

import matplotlib.pyplot as plt

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS
from . import rank_weight


def _plot(fig, **kwargs):
    # One level down, so the method's title sits under the panel letter.
    rank_weight.plot(fig.subfigures(1, 1), **kwargs)


def makeFigure():
    fig = plt.figure(
        figsize=(style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX),
        layout="constrained",
    )
    grid = fig.add_gridspec(1 + len(METHODS), 4)
    panels = {"cccRISE": (fig.add_subfigure(grid[0, 1:3]), {"method": "cccRISE"})}
    for i, method in enumerate(METHODS, start=1):
        for j, annotation in enumerate(ANNOTATIONS):
            panels[f"{method} {annotation}"] = (
                fig.add_subfigure(grid[i, 2 * j : 2 * j + 2]),
                {"method": method, "annotation": annotation},
            )
    for subfig, kwargs in panels.values():
        partial(_plot, **kwargs)(subfig)
    common.label_panels({name: subfig for name, (subfig, _) in panels.items()})
    return fig
