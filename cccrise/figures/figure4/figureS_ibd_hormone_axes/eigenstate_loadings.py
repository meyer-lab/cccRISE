"""C3's and C4's sender (B) and receiver (C) loadings on their eigenstates
(rows as `eigenstate_association`): C3 and C4 share EE sender eigenstates,
weighted differently, and have separate receiver eigenstates."""

import numpy as np
import pandas as pd

from ... import common, style
from .. import _common as S
from . import _common as H

AXES = ("square",)


def plot(axs):
    (ax,) = axs
    rows = H.eigenstate_rows(ax)
    groups = H.eigenstates()
    loadings = np.vstack(
        [
            S.factors()[factor].values[np.array(groups[role]) - 1][:, [2, 3]]
            for role, factor in H.ROLES.items()
        ]
    ).astype(float)
    common.source_data(
        "program_eigenstate_loadings",
        pd.DataFrame(
            loadings,
            columns=["C3 loading", "C4 loading"],
            index=pd.Index(rows, name="Eigenstate"),
        ).assign(Role=[r for r, js in groups.items() for _ in js]),
        "C3 and C4 sender (B; sender rows) and receiver (C; receiver rows) "
        "loadings on each eigenstate either program loads on.",
    )
    ax.imshow(loadings, cmap=style.CMAPS["diverging"], vmin=-1, vmax=1, aspect="auto")
    for (r, c), value in np.ndenumerate(loadings):
        ax.text(
            c,
            r,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            color="white" if abs(value) > 0.6 else style.COLORS["ink"],
        )
    ax.set_xticks(
        [0, 1],
        [f"C{k}\n{top_pair(k)}" for k in (3, 4)],
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.tick_params(length=0)


def top_pair(k: int) -> str:
    D = S.factors()["cp_D"]
    return str(D["lr_pair"].values[np.abs(D.values[:, k - 1]).argmax()]).replace(
        "^", "–"
    )


def makeFigure():
    return common.standalone_fixed(__name__)
