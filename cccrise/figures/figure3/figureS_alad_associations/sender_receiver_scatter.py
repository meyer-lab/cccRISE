"""Per-cell sender vs. receiver weight on the outcome-associated CP
component, with the R² between them across all cells."""

import numpy as np

from ... import common, style
from .._common import COMPONENT, DATASET

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
