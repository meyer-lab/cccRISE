"""NPY1R and GLP1R, the receptors of C3's (PYY-NPY1R) and C4's (GCG-GLP1R)
top LR pairs, on the all-cell map (`cell_map`), stacked beside the receiver
weights (`receiver_maps`). Each is detected in under 4% of any lineage's
cells, so it is capped at its 99% quantile among the cells detecting it."""

from functools import partial

from ... import common
from .. import _common as S
from . import _common as H

AXES = ("square",) * 2
GENES = ["NPY1R", "GLP1R"]


def plot(axs):
    common.source_data(
        "receptor_expression_on_all_cell_map",
        S.programs("cell_map")[["PaCMAP 1", "PaCMAP 2", *GENES]],
        "Normalized NPY1R and GLP1R expression of the cells on the all-cell map.",
    )
    H.all_cell_maps(
        axs, GENES, partial(S.expression_map, detected_only=True), "Expression"
    )


def makeFigure():
    return common.standalone_fixed(__name__)
