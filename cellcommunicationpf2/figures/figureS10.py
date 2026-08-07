"""
Figure S10: CCC-RISE on BALF alad data. Showing weighted sender and receiver cell factors.
"""

from .common import ALAD_STABLE_PATH
from .commonFuncs.plotPaCMAP import make_wc_pacmap_figure


def makeFigure():
    return make_wc_pacmap_figure(ALAD_STABLE_PATH, grid=(6, 6), figsize=(18, 18))

    # for i in range(cp_rank):
    #     plot_wc_per_celltype(X, i + 1, ax[i], cellType="cell_type", factor_matrix="B")
    #     ax[i].set_title(f"Cmp.{i+1} Sender Cells")

    # for i in range(cp_rank):
    #     plot_wc_per_celltype(X, i + 1, ax[i+cp_rank], cellType="cell_type", factor_matrix="C")
    #     ax[i+cp_rank].set_title(f"Cmp.{i+1} Receiver Cells")
