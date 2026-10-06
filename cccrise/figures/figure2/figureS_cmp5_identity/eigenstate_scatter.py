"""Per-cell projection onto the RISE eigenstates behind the granularity
component's sender (x) and receiver (y) loadings, colored by annotated cell
type."""

from ... import common
from .. import _common as S

HIGHLIGHT_CELLTYPE = "Epithelial cells"
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    adata = common.get_cell_projections(S.GRANULARITY_DATASET)
    common.plot_celltype_scatter(
        adata,
        S.GRANULARITY_DATASET,
        adata.obsm["projections"][:, S.GRANULARITY_SENDER_EIGENSTATE - 1],
        adata.obsm["projections"][:, S.GRANULARITY_RECEIVER_EIGENSTATE - 1],
        ax,
        highlight=(HIGHLIGHT_CELLTYPE,),
    )
    ax.set(
        xlabel=f"Eigenstate {S.GRANULARITY_SENDER_EIGENSTATE} projection",
        ylabel=f"Eigenstate {S.GRANULARITY_RECEIVER_EIGENSTATE} projection",
    )
    common.legend(ax, "right", handles=common.celltype_handles(S.GRANULARITY_DATASET))


def makeFigure():
    return common.standalone_fixed(__name__)
