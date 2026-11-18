"""Each cell's projection onto the RISE eigen-states with CPD component 6's
largest sender and receiver loadings."""

import pandas as pd

from ... import common, style
from .. import _common as S

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(S.GRANULARITY_DATASET)
    projections = adata.obsm["projections"]
    x = projections[:, S.GRANULARITY_SENDER_EIGENSTATE - 1]
    y = projections[:, S.GRANULARITY_RECEIVER_EIGENSTATE - 1]
    common.source_data(
        "rise_eigenstate_projection_per_cell",
        pd.DataFrame(
            {
                f"Eigen-state {S.GRANULARITY_SENDER_EIGENSTATE} projection": x,
                f"Eigen-state {S.GRANULARITY_RECEIVER_EIGENSTATE} projection": y,
            },
            index=adata.obs_names.rename("Cell barcode"),
        ),
        "One row per cell: its projection onto the two RISE eigen-states with the "
        "granularity component's largest sender (x) and receiver (y) loadings.",
    )
    ax.scatter(
        x,
        y,
        c=style.COLORS["emphasis"],
        s=style.POINT_SIZE_BACKGROUND,
        linewidths=0,
        rasterized=True,
    )
    common.reference_line(ax, 0, False)
    common.reference_line(ax, 0)
    ax.set(
        xlabel=f"Eigen-state {S.GRANULARITY_SENDER_EIGENSTATE} projection",
        ylabel=f"Eigen-state {S.GRANULARITY_RECEIVER_EIGENSTATE} projection",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
