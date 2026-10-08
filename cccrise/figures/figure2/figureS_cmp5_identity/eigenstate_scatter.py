"""Per-cell projection onto the RISE eigenstates behind the granularity
component's sender (x) and receiver (y) loadings, with dashed lines at each
projection's `common.STRONG_CELL_PERCENTILE`."""

import numpy as np

from ... import common, style
from .. import _common as S

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    projections = common.get_cell_projections(S.GRANULARITY_DATASET).obsm["projections"]
    x = projections[:, S.GRANULARITY_SENDER_EIGENSTATE - 1]
    y = projections[:, S.GRANULARITY_RECEIVER_EIGENSTATE - 1]
    ax.scatter(
        x,
        y,
        c=style.COLORS["emphasis"],
        s=style.POINT_SIZE_BACKGROUND,
        linewidths=0,
        rasterized=True,
    )
    common.reference_line(ax, np.percentile(x, common.STRONG_CELL_PERCENTILE), False)
    common.reference_line(ax, np.percentile(y, common.STRONG_CELL_PERCENTILE))
    ax.set(
        xlabel=f"Eigenstate {S.GRANULARITY_SENDER_EIGENSTATE} projection",
        ylabel=f"Eigenstate {S.GRANULARITY_RECEIVER_EIGENSTATE} projection",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
