"""Cell category x the hormone programs' eigenstates: each category's mean
RISE projection (as `figureS_celltype_eigenstates` for ALAD), from
`run_ibd_programs.eigenstate_celltypes`. The sender eigenstates are EE
cells', the receiver eigenstates EC and tuft cells'."""

import numpy as np

from ... import common, style
from .. import _common as S
from . import _common as H

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    rows = H.eigenstate_rows(ax)
    table = S.programs("eigenstate_celltypes")
    table = table[table["Eigenstate"].isin(rows)]
    common.source_data(
        "eigenstate_category_association",
        table,
        "Per eigen-state the hormone programs (C3, C4) load on and cell category: "
        "cells, their mean RISE projection, and their share of the eigen-state's "
        "squared projection over all cells.",
    )
    mean = table.pivot(
        index="Eigenstate", columns="Category", values="Mean projection"
    ).loc[rows, H.CATEGORIES]
    limit = np.abs(mean.to_numpy()).max()
    image = ax.imshow(
        mean.to_numpy(),
        cmap=style.CMAPS["diverging"],
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(
        range(len(H.CATEGORIES)),
        H.CATEGORIES,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.tick_params(length=0)
    H.label_roles(ax)
    common.colorbar(ax, image, "Mean projection")


def makeFigure():
    return common.standalone_fixed(__name__)
