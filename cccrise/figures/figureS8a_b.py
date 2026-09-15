"""
Figure S8a-b: Heatmaps of ligand-receptor interaction scores between sender and receiver cells for specific cell types and components in cccRISE on BALF alad data.
"""

from .common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotFactors import plot_interaction_heatmap


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    ccc_rise_cmp = 13

    plot_interaction_heatmap(
        X,
        cmp=ccc_rise_cmp,
        ligand="CD200",
        receptor="CD200R1",
        ax=ax[0],
        sender_celltype="Dendritic Cells",
        receiver_celltype="Dendritic Cells",
        celltype_col="broad_cell_type",
        cmap="rocket",
        vmax=0.015,
        sender_label="Dendritic Cells",
        receiver_label="Dendritic Cells",
    )

    plot_interaction_heatmap(
        X,
        cmp=ccc_rise_cmp,
        ligand="CD200",
        receptor="CD200R1",
        ax=ax[2],
        sender_celltype="Dendritic Cells",
        receiver_celltype="Epithelial cells",
        celltype_col="broad_cell_type",
        cmap="rocket",
        vmax=0.015,
        sender_label="Dendritic Cells",
        receiver_label="Epithelial Cells",
    )

    return f
