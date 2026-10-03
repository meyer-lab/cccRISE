"""Per-cell projection onto the RISE eigenstates behind the granularity
component's sender (x) and receiver (y) loadings."""

from .. import common
from ._common import DATASET, HIGHLIGHT_CELLTYPE, RECEIVER_EIGENSTATE, SENDER_EIGENSTATE

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(DATASET)
    common.plot_celltype_scatter(
        adata,
        DATASET,
        adata.obsm["projections"][:, SENDER_EIGENSTATE - 1],
        adata.obsm["projections"][:, RECEIVER_EIGENSTATE - 1],
        ax,
        highlight=(HIGHLIGHT_CELLTYPE,),
    )
    ax.set(
        xlabel=f"Eigenstate {SENDER_EIGENSTATE} projection",
        ylabel=f"Eigenstate {RECEIVER_EIGENSTATE} projection",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
