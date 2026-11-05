"""Figure 5E's annotated types on the zoomed EE/EC/tuft map, then NTS, SCT
and CCK expression on it as Figure 5H draws PYY and GCG: the villus-like
state's hormones besides PYY (`hormone_state_genes`)."""

from ... import common
from .. import _common as S
from ..ee_state_map import type_map
from . import _common as H

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
    H.zoomed_maps(axs[1:], GENES, H.expression, S.EXPRESSION_LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
