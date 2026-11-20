"""Annotated cell type composition of the sender, receiver, and rest groups of
CPD component 6."""

import pandas as pd

from ...runners.datasets import DATASETS
from .. import common, style
from . import _common as S

AXES = ("short",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(S.GRANULARITY_DATASET)
    fractions = pd.crosstab(
        S.cell_groups().to_numpy(),
        adata.obs[DATASETS[S.GRANULARITY_DATASET].celltype_column]
        .astype(str)
        .to_numpy(),
        normalize="index",
    ).reindex(
        index=list(S.GROUPS),
        columns=style.CELL_TYPES[S.GRANULARITY_DATASET],
        fill_value=0,
    )
    common.source_data("cell_weights_groups", S.cell_table(), S.CELL_TABLE_DESCRIPTION)
    common.source_data(
        "celltype_fraction_by_group",
        fractions.rename_axis(index="Group", columns=None),
        "Fraction of each group's cells in each annotated cell type.",
    )
    left = 0
    for celltype, color in style.CELL_TYPE_COLORS[S.GRANULARITY_DATASET].items():
        ax.barh(
            range(len(S.GROUPS)),
            fractions[celltype],
            left=left,
            color=color,
            height=0.7,
        )
        left = left + fractions[celltype]
    ax.set_yticks(range(len(S.GROUPS)), [g.capitalize() for g in S.GROUPS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells")
    common.legend(
        ax,
        "right",
        handles=common.celltype_handles(S.GRANULARITY_DATASET),
        ncols=2,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
