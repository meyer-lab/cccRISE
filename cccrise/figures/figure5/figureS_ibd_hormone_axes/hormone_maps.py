"""Figure 5E's annotated types on the zoomed EE/EC/tuft map, then NTS, SCT
and CCK expression on it: the villus-like state's hormones besides PYY
(`hormone_state_genes`)."""

from ... import common
from .. import _common as S
from ..ee_state_map import type_map

AXES = ("square",) * 4
GENES = ["NTS", "SCT", "CCK"]


def plot(axs):
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    cells = cells.assign(Group=S.map_group(cells["Group"]))
    common.source_data(
        "hormone_expression_on_map",
        cells[["Group", "PaCMAP 1", "PaCMAP 2", *GENES]],
        "Annotated type and normalized expression of each hormone in EE, EC and "
        "tuft cells, on the zoomed map.",
    )
    type_map(axs[0], cells)
    xy = cells[["PaCMAP 1", "PaCMAP 2"]].to_numpy()
    for ax, gene in zip(axs[1:], GENES, strict=True):
        points = S.expression_map(ax, xy, cells[gene].to_numpy())
        S.map_axes(ax, gene)
    common.colorbar(axs[-1], points, "Expression")


def makeFigure():
    return common.standalone_fixed(__name__)
