"""GCG and PYY, the ligands of C4's and C3's top LR pairs, in every annotated
EE cell: a subset of EE cells expresses them, mostly both."""

import numpy as np
import pandas as pd

from ... import common, style
from .. import _common as S

AXES = ("square",)
GENES = ("PYY", "GCG")  # x, y
EE = "Enteroendocrine cells"


def plot(axs):
    (ax,) = axs
    cells = S.programs("enteroendocrine_cells")
    cells = cells.loc[cells["Cell type"] == EE, ["Biopsy", *GENES]]
    common.source_data(
        "ee_gcg_pyy",
        cells,
        "Normalized GCG and PYY expression of every annotated EE cell, with the "
        "fraction detecting neither, one or both.",
    )
    x, y = (cells[g].to_numpy() for g in GENES)
    ax.scatter(
        x,
        y,
        color=style.COLORS["up"],
        s=style.POINT_SIZE_BACKGROUND,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )
    detected = pd.crosstab(x > 0, y > 0, normalize=True)
    pad = 0.1 * np.array([x.max(), y.max()])
    for (dx, dy), (hx, hy) in {
        (False, False): ("left", "bottom"),
        (True, False): ("right", "bottom"),
        (False, True): ("left", "top"),
        (True, True): ("right", "top"),
    }.items():
        ax.text(
            x.max() if dx else pad[0],
            y.max() if dy else pad[1],
            f"{detected.loc[dx, dy]:.0%}",
            ha=hx,
            va=hy,
            fontsize=style.FONT_SIZE_DENSE,
            color=style.COLORS["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )
    ax.set_xlabel(f"{GENES[0]} expression")
    ax.set_ylabel(f"{GENES[1]} expression")
    ax.set_title(f"EE cells (n={len(cells):,})", fontsize=style.FONT_SIZE_LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
