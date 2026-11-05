"""Where Figure 5's zoomed EE/EC/tuft map sits among all cells: a PaCMAP of
every cell's RISE projections (`run_ibd_programs.cell_map`: every EE, EC and
tuft cell, and a seeded draw of the others), colored by lineage, each lineage
(bold) and each zoomed type named at its median position."""

import pandas as pd
from adjustText import adjust_text

from ... import common, style
from .. import _common as S

AXES = ("large_square",)
POINT = 0.2
#: The zoomed map's fine types, as `S.MAP_GROUPS` names them
ZOOMED = {
    "Enteroendocrine cells": "EE",
    "Enterochromaffin cells": "EC",
    "Tuft cells": "Tuft",
}


def plot(axs):
    (ax,) = axs
    cells = S.programs("cell_map")[["Cell type", "Lineage", "PaCMAP 1", "PaCMAP 2"]]
    common.source_data(
        "all_cell_map",
        cells,
        "PaCMAP of every cell's RISE projections (random_state 42): every EE, EC and "
        "tuft cell and a seeded draw of the other cells, with fine type and lineage.",
    )
    lineages = sorted(cells["Lineage"].unique())
    colors = dict(zip(lineages, style.CATEGORICAL, strict=False))
    ax.scatter(
        cells["PaCMAP 1"],
        cells["PaCMAP 2"],
        c=cells["Lineage"].map(colors).to_numpy(),
        s=POINT,
        linewidths=0,
        alpha=0.5,
        rasterized=True,
    )
    xy = ["PaCMAP 1", "PaCMAP 2"]
    zoomed = cells[cells["Cell type"].isin(ZOOMED)]
    centers = pd.concat(
        [
            cells.groupby("Lineage")[xy].median().assign(bold=True),
            zoomed.groupby(zoomed["Cell type"].map(ZOOMED))[xy]
            .median()
            .assign(bold=False),
        ]
    )
    texts = [
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if bold else "normal",
        )
        for name, (x, y, bold) in centers.iterrows()
    ]
    adjust_text(
        texts,
        target_x=centers["PaCMAP 1"].to_numpy(),
        target_y=centers["PaCMAP 2"].to_numpy(),
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": style.COLORS["ink"], "lw": 0.3},
        time_lim=10,
    )
    S.map_axes(ax, xlabel="PaCMAP 1 (all cells)")


def makeFigure():
    return common.standalone_fixed(__name__)
