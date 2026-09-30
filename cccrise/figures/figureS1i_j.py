"""
Figure S1i_j: Heatmaps of ligand-receptor expression products for selected cell types and components.

TODO(IBD): NEEDS CHANGING OR REMOVAL. This figure was written for the retired
BALF cohort, which the IBD cohort supersedes. Its data loader now points at
IBD, but the component, LR pair, cell types and color limits below were picked
for the old cohort and do not carry over: re-pick them for IBD, or delete this
figure.
"""

from .common import (
    get_ibd_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotFactors import plot_interaction_heatmap


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = get_ibd_data(project_cells=True)
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
