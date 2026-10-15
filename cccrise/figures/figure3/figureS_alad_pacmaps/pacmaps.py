"""Every cell's PaCMAP embedding, colored by cell type and by its sample's
6-month outcome."""

import numpy as np
from matplotlib.patches import Patch

from ... import common, style
from .._common import DATASET, OUTCOME, association

AXES = ("square",) * 2


def plot(axs):
    # PaCMAP coordinates have no units: no ticks or spines (first, so the
    # legends are placed without tick labels)
    for ax in axs:
        ax.set(xticks=[], yticks=[])
        ax.spines[:].set_visible(False)
    adata = common.get_cell_projections(DATASET)
    x, y = np.asarray(adata.obsm["PaCMAP"]).T
    common.plot_celltype_scatter(adata, DATASET, x, y, axs[0], thresholds=False)
    axs[0].set_xlabel("Cell type")
    common.legend(
        axs[0],
        "below",
        handles=common.celltype_handles(DATASET),
        fontsize=style.FONT_SIZE_DENSE,
    )
    _outcome_scatter(axs[1], x, y, adata.obs[association(OUTCOME)["group_column"]])


def _outcome_scatter(ax, x, y, outcome) -> None:
    """Cells colored by outcome level, in the component box plot's colors,
    drawn in random order so no level hides the others."""
    levels = common.ordered_levels(outcome.dropna())
    colors = dict(zip(levels, style.CATEGORICAL, strict=False))
    order = np.random.default_rng(0).permutation(len(outcome))
    labels = outcome.to_numpy()[order]
    ax.scatter(
        x[order],
        y[order],
        c=[colors.get(level, style.COLORS["muted"]) for level in labels],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("6-month outcome")
    common.legend(
        ax,
        "below",
        handles=[Patch(color=colors[level], label=level) for level in levels],
        ncols=1,
        fontsize=style.FONT_SIZE_DENSE,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
