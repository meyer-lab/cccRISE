"""
Figure S4e-i: Heatmaps of ligand-receptor expression products for NK cell to Dendritic Cell interactions in CCC-RISE on BALF alad data.
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
    ccc_rise_cmp = 3

    pairs = [["PTN", "PTPRZ1"], ["PTN", "SDC1"]]
    for i, (lig, rec) in enumerate(pairs):
        plot_interaction_heatmap(
            X,
            cmp=ccc_rise_cmp,
            ligand=lig,
            receptor=rec,
            ax=ax[i],
            sender_celltype="Epithelial",
            receiver_celltype="Epithelial",
            sender_label="Sender Epithelial Cells",
            receiver_label="Receiver Epithelial Cells",
        )

    return f
