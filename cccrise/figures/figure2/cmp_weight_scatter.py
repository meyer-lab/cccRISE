"""Per-cell sender vs. receiver weight for the granularity CP component."""

from .. import common, style
from ._common import DATASET, GRANULARITY_COMPONENT, HIGHLIGHT_CELLTYPE


def plot(fig):
    ax = fig.subplots()
    adata = common.get_cell_projections(DATASET)
    common.plot_celltype_scatter(
        adata,
        DATASET,
        adata.obsm["sc_B"][:, GRANULARITY_COMPONENT - 1],
        adata.obsm["rc_C"][:, GRANULARITY_COMPONENT - 1],
        ax,
        highlight=HIGHLIGHT_CELLTYPE,
    )
    ax.set(
        xlabel=f"Cmp. {GRANULARITY_COMPONENT} sender weight",
        ylabel=f"Cmp. {GRANULARITY_COMPONENT} receiver weight",
        title=f"Cmp. {GRANULARITY_COMPONENT} per-cell weights",
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
