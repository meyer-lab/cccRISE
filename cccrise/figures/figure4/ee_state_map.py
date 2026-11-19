"""PaCMAP of the RISE projections of enteroendocrine (EE), enterochromaffin
(EC), and tuft cells, colored by annotated cell type and by each cell's sender
weight on CPD components 3 and 4."""

from .. import common, style
from . import _common as S

AXES = ("square",) * 3
PACMAP_X = "PaCMAP 1"


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
    type_map(axs[0], cells)
    for ax, k in zip(axs[1:], (3, 4), strict=True):
        points = S.weight_map(ax, xy, cells[f"C{k} sender weight"].to_numpy())
        S.map_axes(ax, f"C{k} sender weight", xlabel=PACMAP_X)
    common.colorbar(axs[-1], points, "Weight")


def type_map(ax, cells) -> None:
    """The zoomed map's cells colored by annotated type (`S.MAP_GROUPS`, from
    `cells`' mapped "Group"), each type named at its median in its color."""
    ax.scatter(
        cells["PaCMAP 1"],
        cells["PaCMAP 2"],
        c=cells["Group"].map(S.MAP_GROUPS).to_numpy(),
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    centers = cells.groupby("Group")[["PaCMAP 1", "PaCMAP 2"]].median()
    for group, (x, y) in centers.loc[list(S.MAP_GROUPS)].iterrows():
        ax.text(
            x,
            y,
            group,
            ha="center",
            va="center",
            color=S.MAP_GROUPS[group],
            fontsize=style.FONT_SIZE_LABEL,
            fontweight="bold",
            path_effects=S.LABEL_HALO,
        )
    S.map_axes(ax, xlabel=PACMAP_X)


def makeFigure():
    return common.standalone_fixed(__name__)
