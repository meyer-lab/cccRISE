"""
Figure S4e-i: Heatmaps of ligand-receptor expression products for selected cell types.

TODO(IBD): NEEDS CHANGING OR REMOVAL. This figure was written for the retired
BALF cohort, which the IBD cohort supersedes. Its data loader now points at
IBD, but the component, LR pairs and cell types below were picked for the old
cohort and do not carry over: re-pick them for IBD, or delete this figure.
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
