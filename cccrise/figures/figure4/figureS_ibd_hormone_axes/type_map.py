"""Figure 4E's annotated types on the zoomed EE/EC/tuft map, below the
all-cell map (`cell_map`) it zooms into."""

from ... import common
from .. import _common as S
from .. import ee_state_map

AXES = ("square",)


def plot(axs):
    cells = S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])
    cells = cells.assign(Group=S.map_group(cells["Group"]))
    common.source_data(
        "zoomed_cell_types",
        cells[["Group", "PaCMAP 1", "PaCMAP 2"]],
        "Annotated type of each EE, EC and tuft cell, on the zoomed map.",
    )
    ee_state_map.type_map(axs[0], cells)
    # Figure 4E drops the zoom note its own row explains; here it sits under maps
    axs[0].set_xlabel(f"PaCMAP 1 ({S.ZOOM_NOTE})")


def makeFigure():
    return common.standalone_fixed(__name__)
