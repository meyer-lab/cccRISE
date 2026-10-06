"""Annotated cell type composition of each group: the only panel using cell
type labels, as a check on the label-free groups."""

import pandas as pd

from ...runners.datasets import DATASETS
from .. import common, style

AXES = ("short",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(common.GRANULARITY_DATASET)
    fractions = pd.crosstab(
        common.cell_groups().to_numpy(),
        adata.obs[DATASETS[common.GRANULARITY_DATASET].celltype_column]
        .astype(str)
        .to_numpy(),
        normalize="index",
    ).reindex(
        index=list(common.GROUPS),
        columns=style.CELL_TYPES[common.GRANULARITY_DATASET],
        fill_value=0,
    )
    left = 0
    for celltype, color in style.CELL_TYPE_COLORS[common.GRANULARITY_DATASET].items():
        ax.barh(
            range(len(common.GROUPS)),
            fractions[celltype],
            left=left,
            color=color,
            height=0.7,
        )
        left = left + fractions[celltype]
    ax.set_yticks(range(len(common.GROUPS)), [g.capitalize() for g in common.GROUPS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells")
    ax.tick_params(axis="y", length=0)
    common.legend(
        ax,
        "right",
        handles=common.celltype_handles(common.GRANULARITY_DATASET),
        ncols=2,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
