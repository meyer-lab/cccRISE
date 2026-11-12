"""Every cell's PaCMAP embedding colored by its sender and receiver weights
on the outcome-associated CP component, and by its projection onto RISE
eigenstate `EIGENSTATE`."""

import numpy as np

from .. import common, style
from ..figure3._common import COMPONENT, DATASET

EIGENSTATE = 16
#: Values beyond this percentile of |value| saturate the color scale.
SATURATION_PERCENTILE = 99
AXES = ("square",) * 3


def plot(axs):
    adata = common.get_cell_projections(DATASET)
    x, y = np.asarray(adata.obsm["PaCMAP"]).T
    panels = (
        (
            adata.obsm["sc_B"][:, COMPONENT - 1],
            f"Cmp. {COMPONENT} sender weight",
            "Weight",
        ),
        (
            adata.obsm["rc_C"][:, COMPONENT - 1],
            f"Cmp. {COMPONENT} receiver weight",
            "Weight",
        ),
        (
            adata.obsm["projections"][:, EIGENSTATE - 1],
            f"Eigen-state {EIGENSTATE} projection",
            "Projection",
        ),
    )
    for ax, (values, xlabel, cbar_label) in zip(axs, panels, strict=True):
        points = plot_value_scatter(ax, x, y, values)
        # PaCMAP coordinates have no units
        ax.set(xticks=[], yticks=[], xlabel=xlabel)
        ax.spines[:].set_visible(False)
        common.colorbar(ax, points, cbar_label)


def plot_value_scatter(
    ax, x, y, values, saturation_percentile: float = SATURATION_PERCENTILE
):
    """Cells colored by `values` on the diverging scale, saturating at
    `saturation_percentile`, the largest magnitudes drawn on top."""
    vmax = float(np.percentile(np.abs(values), saturation_percentile))
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


def makeFigure():
    return common.standalone_fixed(__name__)
