"""Per-cell sender vs. receiver weight on the outcome-associated CP
component, colored by each cell's projection onto `EIGENSTATE` on the
lymphocyte blow-up's scale, with the R² between them across all cells."""

import numpy as np
import pandas as pd

from .. import common
from ._common import COMPONENT, DATASET, EIGENSTATE, SPEC
from .celltype_pacmap import projection_limit

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(DATASET)
    sender = adata.obsm["sc_B"][:, COMPONENT - 1]
    receiver = adata.obsm["rc_C"][:, COMPONENT - 1]
    projection = adata.obsm["projections"][:, EIGENSTATE - 1]
    points = common.plot_value_scatter(ax, sender, receiver, projection)
    points.set_clim(-projection_limit(), projection_limit())
    common.colorbar(ax, points, f"Eigen-state {EIGENSTATE} projection")
    r2 = np.corrcoef(sender, receiver)[0, 1] ** 2
    common.source_data(
        f"weight_component{COMPONENT}_cells",
        pd.DataFrame(
            {
                "Cell": adata.obs_names,
                "Sample": adata.obs[SPEC.condition_column].to_numpy(),
                "Sender weight": sender,
                "Receiver weight": receiver,
                f"Eigen-state {EIGENSTATE} projection": projection,
            }
        ),
        f"One row per cell: its per-cell sender and receiver weight on CP "
        f"component {COMPONENT} (the points), and its projection onto RISE "
        f"eigen-state {EIGENSTATE} (the colors). R² = {r2:.4f} is the squared "
        "Pearson correlation between the two across all cells.",
    )
    ax.text(
        0.03,
        0.97,
        f"$R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.set(
        xlabel=f"Cmp. {COMPONENT} sender weight",
        ylabel=f"Cmp. {COMPONENT} receiver weight",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
