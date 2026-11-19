"""PaCMAP of the RISE projections of all EE, EC, and tuft cells and a random
sample of other cells, colored by lineage.

Where Figure 4's zoomed EE/EC/tuft map sits among all cells: a PaCMAP of
every cell's RISE projections (`run_ibd_programs.cell_map`: every EE, EC and
tuft cell, and a seeded draw of the others), colored by lineage, each lineage
(bold) named on its cluster and each zoomed type beside its median, with a
line to it, every label in its lineage's color."""

import numpy as np
import pandas as pd
from adjustText import adjust_text

from ... import common, style
from .. import _common as S

AXES = ("square",)
POINT = 0.2
#: The zoomed map's fine types, as `S.MAP_GROUPS` names them
ZOOMED = {
    "Enteroendocrine cells": "EE",
    "Enterochromaffin cells": "EC",
    "Tuft cells": "Tuft",
}
AVOID_POINTS = 5_000  # cells sampled for fine labels to steer clear of
FINE_OFFSET = 0.12  # map heights from a fine type's median to its label's start
XY = ["PaCMAP 1", "PaCMAP 2"]
LABEL_STYLE = {
    "ha": "center",
    "va": "center",
    "fontsize": style.FONT_SIZE_DENSE,
    "path_effects": S.LABEL_HALO,
}


def plot(axs):
    (ax,) = axs
    cells = S.programs("cell_map")[["Cell type", "Lineage", *XY]]
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
    coarse = cells.groupby("Lineage")[XY].median()
    names = [label(ax, name, xy, colors[name], True) for name, xy in coarse.iterrows()]
    adjust_text(names, ax=ax, time_lim=10)
    zoomed = cells[cells["Cell type"].isin(ZOOMED)]
    fine = zoomed.groupby(zoomed["Cell type"].map(ZOOMED))
    lineage = fine["Lineage"].agg(lambda s: s.mode()[0])
    centers = fine[XY].median()
    # Fine labels start below their medians (open space under the lineage
    # labels), then are pushed off the cells, each keeping a line to its median
    below = np.array([0, -FINE_OFFSET * np.ptp(cells["PaCMAP 2"])])
    texts = [
        label(ax, name, xy + below, colors[lineage[name]], False)
        for name, xy in centers.iterrows()
    ]
    avoid = cells[XY].sample(AVOID_POINTS, random_state=0).to_numpy()
    adjust_text(
        texts,
        x=avoid[:, 0],
        y=avoid[:, 1],
        objects=names,
        ax=ax,
        expand=(2.0, 2.5),
        time_lim=10,
    )
    for text, xy in zip(texts, centers.to_numpy(), strict=True):
        line_to(ax, text, xy)
    S.map_axes(ax, xlabel="PaCMAP 1 (all cells)")


def label(ax, name: str, xy: pd.Series, color: str, bold: bool):
    """`name` at `xy` in `color`, outlined in white."""
    return ax.text(
        *xy,
        name,
        color=color,
        fontweight="bold" if bold else "normal",
        **LABEL_STYLE,
    )


def line_to(ax, text, center: np.ndarray) -> None:
    """Redraw `text` where `adjust_text` left it, with a line to `center`
    ending at its edge."""
    text.remove()
    ax.annotate(
        text.get_text(),
        center,
        xytext=text.get_position(),
        color=text.get_color(),
        arrowprops={
            "arrowstyle": "-",
            "color": style.COLORS["ink"],
            "lw": style.LINE_WIDTH_THIN,
            "shrinkA": 0,
        },
        **LABEL_STYLE,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
