"""
Figure S3: CCC-RISE on BALF alad data. Showing weighted sender and receiver cell factors.
"""

from .common import ALAD_PATH
from .commonFuncs.plotPaCMAP import make_wc_pacmap_figure


def makeFigure():
    return make_wc_pacmap_figure(ALAD_PATH, grid=(6, 6), figsize=(12, 12))

    # for i in range(cp_rank):
    #     plot_wc_per_celltype(X, i + 1, ax[i], cellType="celltype", factor_matrix="B")
    #     ax[i].set_title(f"Cmp.{i+1} Sender Cells")

    # for i in range(cp_rank):
    #     plot_wc_per_celltype(X, i + 1, ax[i+cp_rank], cellType="celltype", factor_matrix="C")
    #     ax[i+cp_rank].set_title(f"Cmp.{i+1} Receiver Cells")
