"""Each cell's unscaled C3 and C4 receiver weight on the all-cell map
(`cell_map`), stacked: where the hormone programs' receivers sit among all
cells."""

from ... import common
from .. import _common as S
from . import _common as H

AXES = ("square",) * 2
COLUMNS = ["C3 receiver weight", "C4 receiver weight"]


def plot(axs):
    common.source_data(
        "receiver_weights_on_all_cell_map",
        S.programs("cell_map")[["PaCMAP 1", "PaCMAP 2", *COLUMNS]],
        "Unscaled per-cell C3 and C4 receiver weights of the cells on the "
        "all-cell map.",
    )
    H.all_cell_maps(axs, COLUMNS, S.weight_map, "Weight")


def makeFigure():
    return common.standalone_fixed(__name__)
