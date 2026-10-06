"""Per-cell projection onto the RISE eigenstates behind the granularity
component's sender (x) and receiver (y) loadings, colored by annotated cell
type."""

from .. import common

HIGHLIGHT_CELLTYPE = "Epithelial cells"
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(common.GRANULARITY_DATASET)
    common.plot_celltype_scatter(
        adata,
        common.GRANULARITY_DATASET,
        adata.obsm["projections"][:, common.GRANULARITY_SENDER_EIGENSTATE - 1],
        adata.obsm["projections"][:, common.GRANULARITY_RECEIVER_EIGENSTATE - 1],
        ax,
        highlight=(HIGHLIGHT_CELLTYPE,),
    )
    ax.set(
        xlabel=f"Eigenstate {common.GRANULARITY_SENDER_EIGENSTATE} projection",
        ylabel=f"Eigenstate {common.GRANULARITY_RECEIVER_EIGENSTATE} projection",
    )
    common.legend(
        ax, "right", handles=common.celltype_handles(common.GRANULARITY_DATASET)
    )


def makeFigure():
    return common.standalone_fixed(__name__)
