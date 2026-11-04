"""Enteroendocrine (EE), enterochromaffin (EC) and tuft cells on a PaCMAP of
their RISE projections (zoomed: only these three annotated types), by
annotated type, and each cell's unscaled C3 and C4 sender weight. As Figure 2
paints weights on cells."""

from matplotlib.lines import Line2D

from .. import common, style
from . import _common as S

AXES = ("square",) * 3


def plot(axs):
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    cells = cells.assign(Group=S.map_group(cells["Group"]))
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    common.source_data(
        "ee_ec_map",
        cells[
            ["Group", "C3 sender weight", "C4 sender weight", "PaCMAP 1", "PaCMAP 2"]
        ],
        "EE, EC and tuft cells: PaCMAP of their RISE projections (random_state 42), "
        "annotated type and unscaled per-cell C3 and C4 sender weights.",
    )
    axs[0].scatter(
        *xy.T,
        c=cells["Group"].map(S.MAP_GROUPS).to_numpy(),
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    S.map_axes(axs[0])
    common.legend(
        axs[0],
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
            for g, c in S.MAP_GROUPS.items()
        ],
        ncols=2,
        fontsize=style.FONT_SIZE_DENSE,
        handletextpad=0.1,
        columnspacing=0.8,
    )
    for ax, k in zip(axs[1:], (3, 4), strict=True):
        points = S.weight_map(ax, xy, cells[f"C{k} sender weight"].to_numpy())
        S.map_axes(ax, f"C{k} sender weight")
    common.colorbar(axs[-1], points, "Weight")


def makeFigure():
    return common.standalone_fixed(__name__)
