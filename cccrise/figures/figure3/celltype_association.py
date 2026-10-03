"""Each cell type's mean per-cell sender (left) and receiver (right) weight on
every CP component, on one color scale, with the outcome-associated component
outlined (the Shiny explorer's Sender/Receiver Cells panel)."""

import numpy as np
from matplotlib.patches import Rectangle

from ...utils import celltype_component_association
from .. import common, style
from ._common import COMPONENT, DATASET, SPEC

ROLES = ("sender", "receiver")
AXES = ("square",) * len(ROLES)


def plot(axs):
    adata = common.get_cell_projections(DATASET)
    tables = [
        celltype_component_association(
            adata, factor_type=role, celltype_col=SPEC.celltype_column, stat="mean"
        ).reindex(style.CELL_TYPES[DATASET])
        for role in ROLES
    ]
    vmax = max(np.nanmax(np.abs(t.to_numpy())) for t in tables)
    for ax, role, table in zip(axs, ROLES, tables, strict=True):
        image = ax.imshow(
            table.to_numpy(),
            cmap=style.CMAPS["diverging"],
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.add_patch(
            Rectangle(
                (COMPONENT - 1.5, -0.5),
                1,
                len(table),
                fill=False,
                edgecolor="black",
                linewidth=style.LINE_WIDTH,
            )
        )
        ax.set_xticks(range(table.shape[1]), table.columns)
        ax.set_yticks(range(len(table)), table.index)
        ax.set_xlabel(f"CP component\n({role} weight)")
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    # Same cell types in both, so the rows are labeled once
    axs[1].tick_params(axis="y", labelleft=False)
    common.colorbar(axs[-1], image, "Mean cell weight")


def makeFigure():
    return common.standalone_fixed(__name__)
