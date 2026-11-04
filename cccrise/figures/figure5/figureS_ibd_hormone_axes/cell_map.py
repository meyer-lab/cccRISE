"""Where Figure 5's zoomed EE/EC/tuft map sits among all cells: a PaCMAP of
every cell's RISE projections (`run_ibd_programs.cell_map`: every EE, EC and
tuft cell, and a seeded draw of the others), colored by lineage, each fine
cell type named at its median position (the zoomed types in bold)."""

from adjustText import adjust_text
from matplotlib.lines import Line2D

from ... import common, style
from .. import _common as S

AXES = ("large_square",)
POINT = 0.2
ZOOMED = {"Enteroendocrine cells", "Enterochromaffin cells", "Tuft cells"}


def plot(axs):
    (ax,) = axs
    cells = S.programs("cell_map").drop(columns="Group")
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
    centers = cells.groupby("Cell type")[["PaCMAP 1", "PaCMAP 2"]].median()
    texts = [
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if name in ZOOMED else "normal",
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
    common.legend(
        ax,
        "below",
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                color=c,
                markersize=style.MARKER_SIZE,
                label=g,
            )
            for g, c in colors.items()
        ],
        ncols=len(colors),
        fontsize=style.FONT_SIZE_DENSE,
        handletextpad=0.1,
        columnspacing=0.8,
    )
    S.map_axes(ax, xlabel="PaCMAP 1 (all cells)")


def makeFigure():
    return common.standalone_fixed(__name__)
