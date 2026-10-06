"""Agreement between settings: Spearman correlation of ligand scores, over
ligands both priors cover, median across CP components. Settings within a
prior agree; the two priors barely agree under any setting."""

import itertools

import numpy as np
from scipy.stats import spearmanr

from .. import common, style
from . import _common


def plot(fig):
    shared = _common.load().query("universe == 'shared'").dropna(subset=["score"])
    wide = shared.pivot_table(
        index=["component", "ligand"], columns=_common.SETTING, values="score"
    )
    rows = _common.settings()
    rho = np.array(
        [
            [
                np.median(
                    [
                        spearmanr(g[a], g[b], nan_policy="omit")[0]
                        for _, g in wide.groupby(level=0)
                    ]
                )
                for b in rows
            ]
            for a in rows
        ]
    )
    ax = fig.subplots()
    image = ax.imshow(rho, cmap=style.CMAPS["diverging"], vmin=-1, vmax=1)
    for i, j in itertools.product(range(len(rows)), repeat=2):
        ax.text(
            j,
            i,
            f"{rho[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color=style.COLORS["ink"],
        )
    ax.set_xticks(range(len(rows)), range(1, len(rows) + 1))
    ax.set_yticks(
        range(len(rows)),
        [f"{_common.setting_label(*r)}  {i}" for i, r in enumerate(rows, 1)],
    )
    ax.set_title("Ligand-score Spearman between settings")
    fig.colorbar(image, ax=ax, label="median Spearman ρ", shrink=0.8, pad=0.02)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
