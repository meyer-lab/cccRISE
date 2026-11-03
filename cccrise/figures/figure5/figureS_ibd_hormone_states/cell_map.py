"""Where Figure 5's zoomed EE/EC map sits among all cells: a PaCMAP of every
cell's RISE projections (`run_ibd_programs.cell_map`: every EE and EC cell,
and a seeded draw of the others, grey)."""

from matplotlib.lines import Line2D

from ... import common, style
from .. import _common as S

AXES = ("square",)
POINT = 0.05


def plot(axs):
    (ax,) = axs
    cells = S.programs("cell_map")
    common.source_data(
        "all_cell_map",
        cells,
        "PaCMAP of every cell's RISE projections (random_state 42): every EE and EC "
        "cell (EE split by L-hormone detection) and a seeded draw of the other cells.",
    )
    other = (cells["Group"] == "Other").to_numpy()
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    ax.scatter(*xy[other].T, color="0.88", s=POINT, linewidths=0, rasterized=True)
    ax.scatter(
        *xy[~other].T,
        c=cells.loc[~other, "Group"].map(S.EE_GROUPS).to_numpy(),
        s=6 * POINT,
        linewidths=0,
        rasterized=True,
    )
    ax.legend(
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
            for g, c in S.EE_GROUPS.items()
        ],
        loc="upper left",
        fontsize=style.FONT_SIZE_DENSE,
        handletextpad=0.1,
        borderaxespad=0.1,
    )
    S.map_axes(ax, xlabel="PaCMAP 1 (all cells)")


def makeFigure():
    return common.standalone_fixed(__name__)
