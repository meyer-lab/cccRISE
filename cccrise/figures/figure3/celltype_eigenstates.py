"""Cell type x RISE eigenstate association (mean projection), with
`EIGENSTATE`, the eigenstate behind the outcome-associated component's sender
and receiver weights, outlined."""

from collections.abc import Sequence

import anndata
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from ...runners.datasets import DATASETS
from ...utils import celltype_component_association
from .. import common, style
from ._common import DATASET, EIGENSTATE

AXES = ("double_rect",)


def plot(axs):
    (ax,) = axs
    plot_celltype_eigenstate_association(
        common.get_cell_projections(DATASET), DATASET, ax, highlight=(EIGENSTATE,)
    )


def makeFigure():
    return common.standalone_fixed(__name__)


def plot_celltype_eigenstate_association(
    adata: anndata.AnnData,
    dataset: str,
    ax: plt.Axes,
    highlight: Sequence[int] = (),
) -> None:
    """Cell type (rows, `style.CELL_TYPES` order) x RISE eigenstate mean
    projection (mean over the cell type's cells of each eigenstate projection),
    with the `highlight` eigenstates' columns outlined by dashed boxes."""
    table = celltype_component_association(
        adata,
        factor_type="eigenstate",
        celltype_col=DATASETS[dataset].celltype_column,
        stat="mean",
    ).reindex(style.CELL_TYPES[dataset])
    vmax = np.nanmax(np.abs(table.to_numpy()))
    image = ax.imshow(
        table.to_numpy(),
        cmap=style.CMAPS["diverging"],
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for eigenstate in highlight:
        column = table.columns.get_loc(eigenstate)
        ax.add_patch(
            Rectangle(
                (column - 0.5, -0.5),
                1,
                len(table),
                fill=False,
                linestyle=style.LINE_STYLE_REFERENCE,
                edgecolor=style.COLORS["ink"],
                linewidth=style.LINE_WIDTH,
                clip_on=False,
            )
        )
    ax.set_xlabel("RISE eigen-state")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    common.colorbar(ax, image, "Mean projection")
