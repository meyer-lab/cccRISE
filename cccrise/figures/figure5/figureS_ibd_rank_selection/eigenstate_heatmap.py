"""Cell category x every RISE eigenstate: each category's mean projection
(as `figureS_celltype_eigenstates` for ALAD), from
`run_ibd_programs.eigenstate_celltypes`; the hormone programs' sender and
receiver eigenstates (`_common.eigenstates`) labeled bold in the sender and
receiver colors."""

import numpy as np

from ... import common, style
from .. import _common as S
from ..figureS_ibd_hormone_axes import _common as H

AXES = ("full_labeled",)


def plot(axs):
    (ax,) = axs
    table = S.programs("eigenstate_celltypes")
    common.source_data(
        "eigenstate_category_association_all",
        table,
        "Per RISE eigenstate and cell category: cells, their mean RISE projection, "
        "and their share of the eigenstate's squared projection over all cells.",
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
    ax.set_xticks(
        range(mean.shape[1]), mean.columns, rotation=90, fontsize=style.FONT_SIZE_DENSE
    )
    ax.set_yticks(range(len(mean)), mean.index, fontsize=style.FONT_SIZE_DENSE)
    roles = {j: role for role, js in H.eigenstates().items() for j in js}
    for label in ax.get_xticklabels():
        if (role := roles.get(int(label.get_text()))) is not None:
            label.set(fontweight="bold", color=style.COLORS[role])
    ax.set_xlabel("RISE eigenstate (bold: C3/C4 sender, receiver)")
    ax.tick_params(length=0)
    common.colorbar(ax, image, "Mean projection")


def makeFigure():
    return common.standalone_fixed(__name__)
