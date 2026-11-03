"""Per-cell sender vs. receiver weight on the outcome-associated CP
component, with the R² between them across all cells."""

import numpy as np
import pandas as pd

from ... import common, style
from .._common import COMPONENT, DATASET, SPEC

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(DATASET)
    sender = adata.obsm["sc_B"][:, COMPONENT - 1]
    receiver = adata.obsm["rc_C"][:, COMPONENT - 1]
    ax.scatter(
        sender,
        receiver,
        color=style.COLORS["emphasis"],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    r2 = np.corrcoef(sender, receiver)[0, 1] ** 2
    common.source_data(
        f"weight_component{COMPONENT}_cells",
        pd.DataFrame(
            {
                "Cell": adata.obs_names,
                "Sample": adata.obs[SPEC.condition_column].to_numpy(),
                "Sender weight": sender,
                "Receiver weight": receiver,
            }
        ),
        f"One row per cell: its per-cell sender and receiver weight on CP "
        f"component {COMPONENT} (the points). R² = {r2:.4f} is the squared "
        "Pearson correlation between the two across all cells.",
    )
    ax.text(
        0.03,
        0.97,
        f"$R^2$ = {r2:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.set(
        xlabel=f"Cmp. {COMPONENT} sender weight",
        ylabel=f"Cmp. {COMPONENT} receiver weight",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
