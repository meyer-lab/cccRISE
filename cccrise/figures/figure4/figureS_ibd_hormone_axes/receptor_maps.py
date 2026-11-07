"""NPY1R and GLP1R, the receptors of C3's (PYY-NPY1R) and C4's (GCG-GLP1R)
top LR pairs, on Figure 4's zoomed map as Figure 4H draws its ligands,
stacked beside the receiver weights (`receiver_maps`)."""

from ... import common
from .. import _common as S
from . import _common as H

AXES = ("square",) * 2
GENES = ["NPY1R", "GLP1R"]


def plot(axs):
    common.source_data(
        "receptor_expression_on_map",
        S.programs("enteroendocrine_cells").dropna(subset=["PaCMAP 1"])[
            ["PaCMAP 1", "PaCMAP 2", *GENES]
        ],
        "Normalized NPY1R and GLP1R expression of EE, EC and tuft cells, on the "
        "zoomed map.",
    )
    H.zoomed_maps(axs, GENES, H.expression, S.EXPRESSION_LABEL)


def makeFigure():
    return common.standalone_fixed(__name__)
