"""Cell type x RISE eigenstate association (mean projection), with the
eigenstates behind Figure 2's granularity component highlighted."""

from collections.abc import Sequence

import anndata
import numpy as np
from matplotlib import pyplot as plt

from ...runners.datasets import DATASETS
from ...utils import celltype_component_association
from .. import common, style
from ..figure2 import _common as S


def plot(fig):
    ax = fig.subplots()
    plot_celltype_eigenstate_association(
        common.get_cell_projections(S.GRANULARITY_DATASET),
        S.GRANULARITY_DATASET,
        ax,
        highlight=(
            S.GRANULARITY_SENDER_EIGENSTATE,
            S.GRANULARITY_RECEIVER_EIGENSTATE,
        ),
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 60 * style.MM))


def plot_celltype_eigenstate_association(
    adata: anndata.AnnData,
    dataset: str,
    ax: plt.Axes,
    highlight: Sequence[int] = (),
) -> None:
    """Cell type (rows, `style.CELL_TYPES` order) x RISE eigenstate mean
    projection (mean over the cell type's cells of each eigenstate projection),
    with the `highlight` eigenstates' labels bold."""
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
    for label in ax.get_xticklabels():
        if int(label.get_text()) in highlight:
            label.set_fontweight("bold")
    ax.set_xlabel("RISE eigenstate")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.figure.colorbar(image, ax=ax, label="Mean projection", shrink=0.8)
