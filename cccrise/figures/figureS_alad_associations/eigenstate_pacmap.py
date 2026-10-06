"""Every cell's PaCMAP embedding colored by its projection onto RISE
eigenstate `EIGENSTATE`."""

import numpy as np

from .. import common
from ..figure3._common import DATASET
from ..figure3.pacmaps import _saturation, _value_scatter

EIGENSTATE = 16
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(DATASET)
    x, y = np.asarray(adata.obsm["PaCMAP"]).T
    values = adata.obsm["projections"][:, EIGENSTATE - 1]
    points = _value_scatter(ax, x, y, values, _saturation(values))
    # PaCMAP coordinates have no units
    ax.set(xticks=[], yticks=[], xlabel=f"Eigenstate {EIGENSTATE} projection")
    ax.spines[:].set_visible(False)
    common.colorbar(ax, points, "Projection")


def makeFigure():
    return common.standalone_fixed(__name__)
