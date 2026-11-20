"""Cell category x every RISE eigenstate: each category's mean projection
(as Figure 3's `celltype_eigenstates` for ALAD), from
`run_ibd_programs.eigenstate_celltypes`, categories as
`figureS_ibd_hormone_axes` breaks the hormone programs' eigenstates down."""

import numpy as np

from ... import common, style
from .. import _common as S
from ..figureS_ibd_hormone_axes import _common as H
from ._common import component_ticks

AXES = ("full_labeled",)


def plot(axs):
    (ax,) = axs
    table = S.programs("eigenstate_celltypes")
    common.source_data(
        "eigenstate_category_association_all",
        table,
        "Per RISE eigen-state and cell category: cells, their mean RISE projection, "
        "and their share of the eigen-state's squared projection over all cells.",
    )
    mean = table.pivot(
        index="Category", columns="Eigenstate", values="Mean projection"
    ).loc[H.CATEGORIES]
    limit = np.abs(mean.to_numpy()).max()
    image = ax.imshow(
        mean.to_numpy(),
        cmap=style.CMAPS["diverging"],
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    component_ticks(ax, mean.shape[1])
    ax.set_yticks(range(len(mean)), mean.index, fontsize=style.FONT_SIZE_DENSE)
    ax.set_xlabel("RISE eigen-state")
    common.colorbar(ax, image, "Mean projection")


def makeFigure():
    return common.standalone_fixed(__name__)
