"""Per-cell sender vs. receiver weight on the outcome-associated CP
component, its predominant cell types enlarged, with the cell type legend
shared by this figure's cell type panels."""

from collections.abc import Sequence

from matplotlib import pyplot as plt

from .. import common, style
from ._common import COMPONENT, COMPONENT_CELLTYPES, DATASET

AXES = ("square",)


def plot(axs):
    plot_component_weight_scatter(
        DATASET, COMPONENT, axs[0], highlight=COMPONENT_CELLTYPES
    )


def makeFigure():
    return common.standalone_fixed(__name__)


def plot_component_weight_scatter(
    dataset: str, component: int, ax: plt.Axes, highlight: Sequence[str] = ()
) -> None:
    """Per-cell sender vs. receiver weight on one CP component
    (`plot_celltype_scatter`), with the cell type legend below it."""
    adata = common.get_cell_projections(dataset)
    common.plot_celltype_scatter(
        adata,
        dataset,
        adata.obsm["sc_B"][:, component - 1],
        adata.obsm["rc_C"][:, component - 1],
        ax,
        highlight=highlight,
    )
    ax.set(
        xlabel=f"Cmp. {component} sender weight",
        ylabel=f"Cmp. {component} receiver weight",
    )
    common.legend(
        ax,
        "below",
        handles=common.celltype_handles(dataset),
        fontsize=style.FONT_SIZE_DENSE,
    )
