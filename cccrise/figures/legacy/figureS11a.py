"""
Figure A3za: XXX
"""

from typing import Literal

import numpy as np
import scanpy as sc
import seaborn as sns

from ...utils import (
    add_obs_cmp_label,
    add_obs_cmp_unique_one,
)
from ..common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = get_alad_data(project_cells=True)
    # Change type to sender or receiver as needed
    # Sender cell type comparisons
    # cmp = (5, "CD4 T cells") # cmp5
    # cmp = (7, "CD4 T cells") # cmp7
    # cmp = (13, "Dendritic Cells") # cmp13
    # cmp = (17, "NK cells") # cmp17
    # type = "sender"
    # Receiver cell type comparisons
    # cmp = (5, "Dendritic Cells") # cmp5
    # cmp = (7, "CD4 T cells") # cmp7
    # cmp = (13, "Epithelial cells") # cmp13
    cmp: tuple[int, str] = (17, "CD8 T cells")  # cmp17
    type = "receiver"

    ccc_rise_cmp = cmp[0]
    pos_perc = [True, False]
    axs = 0

    for i in pos_perc:
        X_epi_sender = X[X.obs["broad_cell_type"] == cmp[1]]
        X_epi_sender = add_obs_cmp_label(
            X_epi_sender, cmp=ccc_rise_cmp, pos=i, top_perc=10, type=type
        )
        X_epi_sender = add_obs_cmp_unique_one(X_epi_sender, cmp=ccc_rise_cmp)

        method_test: list[Literal["logreg", "wilcoxon"]] = ["logreg", "wilcoxon"]

        for method in method_test:
            sc.tl.rank_genes_groups(X_epi_sender, "Label", method=method)
            results = X_epi_sender.uns["rank_genes_groups"]
            top_n = 20
            top_idx = np.argsort(results["scores"][f"Cmp{ccc_rise_cmp}"])[-top_n:][::-1]
            top_genes = results["names"][f"Cmp{ccc_rise_cmp}"][top_idx]
            top_scores = results["scores"][f"Cmp{ccc_rise_cmp}"][top_idx]

            sns.barplot(
                x=top_scores,
                y=top_genes,
                ax=ax[axs],
            )
            ax[axs].set_xlabel(method)
            ax[axs].set_ylabel("Gene")
            axs += 1

    ax[0].set_title("Top 10 perc vs all other cells: logreg")
    ax[1].set_title("Top 10 perc vs all other cells: Wilcoxon")
    ax[2].set_title("Bottom 10 perc vs all other cells: logreg")
    ax[3].set_title("Bottom 10 perc vs all other cells: Wilcoxon")

    return f
