"""Every cell's PaCMAP embedding, colored by cell type, by its sender weight
on the outcome-associated CP component, and by its sample's 6-month
outcome."""

import numpy as np
from matplotlib.patches import Patch

from .. import common, style
from ._common import COMPONENT, DATASET, OUTCOME, association

#: Values beyond this percentile of |value| saturate the color scale.
SATURATION_PERCENTILE = 99
AXES = ("square",) * 3


def plot(axs):
    # PaCMAP coordinates have no units: no ticks or spines (first, so the
    # outcome legend is placed without tick labels)
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

    sender = adata.obsm["sc_B"][:, COMPONENT - 1]
    points = _value_scatter(axs[1], x, y, sender, _saturation(sender))
    axs[1].set_xlabel(f"Cmp. {COMPONENT} sender weight")
    common.colorbar(axs[1], points, "Weight")

    _outcome_scatter(axs[2], x, y, adata.obs[association(OUTCOME)["group_column"]])


def _saturation(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), SATURATION_PERCENTILE))


def _value_scatter(ax, x, y, values, vmax):
    """Cells colored by `values` on the diverging scale, the largest
    magnitudes drawn on top."""
    order = np.argsort(np.abs(values))
    return ax.scatter(
        x[order],
        y[order],
        c=values[order],
        cmap=style.CMAPS["diverging"],
        vmin=-vmax,
        vmax=vmax,
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )


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
