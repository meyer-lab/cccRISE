"""Share of each eigen-state's squared RISE projection carried by each cell
category, for the eigen-states IBD CPD components 3 and 4 load on.

Each of the hormone programs' eigenstates (rows as
`eigenstate_association`) split by the cell categories carrying it: their
share of its squared RISE projection over all cells. The C3 receiver
eigenstate is carried by EC cells, C4's by tuft cells."""

import numpy as np
from matplotlib.patches import Patch

from ... import common
from .. import _common as S
from . import _common as H

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    rows = H.eigenstate_rows(ax)
    table = S.programs("eigenstate_celltypes")
    share = table.pivot(
        index="Eigenstate", columns="Category", values="Squared projection share"
    ).loc[rows]
    share["Immune"] = share[H.IMMUNE].sum(axis=1)
    common.source_data(
        "eigenstate_category_share",
        share[list(H.CATEGORY_COLORS)],
        "Share of each eigen-state's squared RISE projection, over all cells, "
        "carried by each cell category (immune lineages pooled)." + S.PER_CELL_NOTE,
    )
    left = np.zeros(len(rows))
    for category, color in H.CATEGORY_COLORS.items():
        values = share[category].to_numpy()
        ax.barh(range(len(rows)), values, left=left, color=color, height=0.75)
        left += values
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Share of squared\nprojection")
    common.legend(
        ax,
        "right",
        handles=[Patch(color=c, label=n) for n, c in H.CATEGORY_COLORS.items()],
    )


def makeFigure():
    return common.standalone_fixed(__name__)
