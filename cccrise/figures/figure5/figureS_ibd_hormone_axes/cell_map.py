"""Where Figure 5's zoomed EE/EC/tuft map sits among all cells: a PaCMAP of
every cell's RISE projections (`run_ibd_programs.cell_map`: every EE, EC and
tuft cell, and a seeded draw of the others), colored by lineage, each lineage
named at its median position."""

from adjustText import adjust_text

from ... import common, style
from .. import _common as S
from . import _common as H

AXES = ("large_square",)


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
        s=H.MAP_POINT,
        linewidths=0,
        alpha=0.5,
        rasterized=True,
    )
    centers = cells.groupby("Lineage")[["PaCMAP 1", "PaCMAP 2"]].median()
    texts = [
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold",
        )
        for name, (x, y) in centers.iterrows()
    ]
    adjust_text(
        texts,
        target_x=centers["PaCMAP 1"].to_numpy(),
        target_y=centers["PaCMAP 2"].to_numpy(),
        ax=ax,
        arrowprops={"arrowstyle": "-", "color": style.COLORS["ink"], "lw": 0.3},
        time_lim=10,
    )
    S.map_axes(ax, xlabel=H.ALL_CELLS)


def makeFigure():
    return common.standalone_fixed(__name__)
