"""Each cell type's mean per-cell sender (left) and receiver (right) weight on
every CP component, on one color scale, with the outcome-associated component
outlined (the Shiny explorer's Sender/Receiver Cells panel)."""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ...utils import celltype_component_association
from .. import common, style
from ..figure3._common import COMPONENT, DATASET, SPEC

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
    for part, (role, table) in enumerate(zip(ROLES, tables, strict=True)):
        common.source_data(
            f"mean_{role}_weight_by_celltype",
            table.rename_axis(index="Cell type").add_prefix("Component "),
            f"Mean per-cell {role} weight of each annotated cell type's cells on "
            "every CP component.",
            part=part,
        )
    common.source_data(
        "cell_component_weights",
        _cell_weights(adata),
        "Every cell's sender and receiver weight on each CP component, with its "
        "sample and annotated cell type: the values averaged per cell type.",
    )
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
        ax.set_yticks(range(len(table)), table.index, fontsize=style.FONT_SIZE_DENSE)
        ax.set_xlabel(f"CP component\n({role} weight)")
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    # Same cell types in both, so the rows are labeled once
    axs[1].tick_params(axis="y", labelleft=False)
    common.colorbar(axs[-1], image, "Mean cell weight")


def _cell_weights(adata) -> pd.DataFrame:
    weights = {
        f"Cmp {c + 1} {role} weight": adata.obsm[key][:, c]
        for role, key in zip(ROLES, ("sc_B", "rc_C"), strict=True)
        for c in range(adata.obsm[key].shape[1])
    }
    return pd.DataFrame(
        {
            "Cell": adata.obs_names,
            "Sample": adata.obs[SPEC.condition_column].to_numpy(),
            "Cell type": adata.obs[SPEC.celltype_column].to_numpy(),
            **weights,
        }
    )


def makeFigure():
    return common.standalone_fixed(__name__)
