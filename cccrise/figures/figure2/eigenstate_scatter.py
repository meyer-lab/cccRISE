"""Per-cell projection onto the RISE eigenstates behind the granularity
component's sender (x) and receiver (y) loadings, with the shared cell type
legend for this row."""

from .. import common, style
from ._common import DATASET, HIGHLIGHT_CELLTYPE, RECEIVER_EIGENSTATE, SENDER_EIGENSTATE


def plot(fig):
    ax = fig.subplots()
    adata = common.get_cell_projections(DATASET)
    common.plot_celltype_scatter(
        adata,
        DATASET,
        adata.obsm["projections"][:, SENDER_EIGENSTATE - 1],
        adata.obsm["projections"][:, RECEIVER_EIGENSTATE - 1],
        ax,
        highlight=HIGHLIGHT_CELLTYPE,
    )
    ax.set(
        xlabel=f"Eigenstate {SENDER_EIGENSTATE} projection",
        ylabel=f"Eigenstate {RECEIVER_EIGENSTATE} projection",
        title="RISE eigenstate projections",
    )
    common.celltype_legend(ax, DATASET, loc="center left", bbox_to_anchor=(1, 0.5))


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
