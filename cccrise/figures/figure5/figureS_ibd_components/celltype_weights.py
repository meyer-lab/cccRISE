"""Each fine cell type's mean per-cell sender (left) and receiver (right)
weight on every IBD CP component, as Figure 3 shows ALAD's
(`figure3.celltype_association`). Only the `TOP_TYPES` cell types of largest
mean weight per component and role are shown, grouped by lineage; the
source data hold every cell type."""

import pandas as pd

from ... import common, style
from ...figure3.celltype_association import ROLES, draw_weights
from .. import _common as S

TOP_TYPES = 3
#: Wider than Figure 3's squares, so the supplement's two rows fill its width
AXES = ("rect",) * len(ROLES)


def shown_celltypes(weights: pd.DataFrame) -> list[str]:
    """The union of every (role, component)'s `TOP_TYPES` heaviest cell
    types, in lineage order, then alphabetically."""
    top = weights.sort_values("mean_weight", ascending=False)
    top = top.groupby(["role", "component"]).head(TOP_TYPES)
    lineage = {name: i for i, name in enumerate(style.CELL_TYPES[S.DATASET])}
    shown = top.drop_duplicates("celltype")
    return sorted(shown["celltype"], key=lambda c: (lineage[_lineage(weights, c)], c))


def _lineage(weights: pd.DataFrame, celltype: str) -> str:
    return weights.loc[weights["celltype"] == celltype, "lineage"].iloc[0]


def plot(axs):
    weights = S.celltype_weights()
    weights = weights[weights["level"] == "fine"]
    shown = shown_celltypes(weights)
    tables = [
        weights[weights["role"] == role]
        .pivot(index="celltype", columns="component", values="mean_weight")
        .reindex(shown)
        for role in ROLES
    ]
    common.source_data(
        "mean_weight_by_celltype",
        weights[
            ["role", "celltype", "lineage", "n_cells", "component", "mean_weight"]
        ].rename(
            columns={
                "role": "Role",
                "celltype": "Cell type",
                "lineage": "Lineage",
                "n_cells": "Cells",
                "component": "CP component",
                "mean_weight": "Mean cell weight",
            }
        ),
        "Mean per-cell sender and receiver weight of every annotated fine cell "
        f"type on every CP component; the panel shows the {TOP_TYPES} heaviest "
        "cell types of each component and role.",
    )
    draw_weights(axs, tables)


def makeFigure():
    return common.standalone_fixed(__name__)
