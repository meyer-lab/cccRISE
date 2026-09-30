"""
Figure S1j-k: Heatmaps of ligand-receptor expression products for selected cell types and components in cccRISE on BALF COVID-19 data.
"""

from .common import (
    get_covid_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotFactors import plot_interaction_heatmap


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = get_covid_data(project_cells=True)
    ccc_rise_cmp = 6

    plot_interaction_heatmap(
        X,
        cmp=ccc_rise_cmp,
        ligand="CCL19",
        receptor="CCR7",
        ax=ax[0],
        sender_celltype="Macrophages",
        receiver_celltype="Macrophages",
        vmax=0.12,
        sender_label="Sender Cells (Macrophages)",
        receiver_label="Receiver Cells (Macrophages)",
    )

    plot_interaction_heatmap(
        X,
        cmp=ccc_rise_cmp,
        ligand="CCL19",
        receptor="CCR7",
        ax=ax[1],
        sender_celltype="Macrophages",
        receiver_celltype="NK",
        vmax=0.12,
        sender_label="Sender Cells (Macrophages)",
        receiver_label="Receiver Cells (NK cells)",
    )

    return f
