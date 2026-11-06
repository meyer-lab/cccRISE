"""Each EE, EC and tuft cell's unscaled C3 and C4 receiver weight on Figure
5's zoomed map, stacked: C3 receives on EC cells, C4 on tuft cells."""

from ... import common
from .. import _common as S
from . import _common as H

AXES = ("square",) * 2
COLUMNS = ["C3 receiver weight", "C4 receiver weight"]


def plot(axs):
    common.source_data(
        "receiver_weights_on_map",
        S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])[
            ["PaCMAP 1", "PaCMAP 2", *COLUMNS]
        ],
        "Unscaled per-cell C3 and C4 receiver weights of EE, EC and tuft cells, "
        "on the zoomed map.",
    )
    H.zoomed_maps(axs, COLUMNS, S.weight_map, "Weight")


def makeFigure():
    return common.standalone_fixed(__name__)
