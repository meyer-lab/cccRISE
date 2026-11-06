"""The IBD CP factor matrices, as Figure 2 draws ALAD's
(`figure2.cp_factor_matrices`)."""

from .. import common
from ..figure2.cp_factor_matrices import AXES, draw  # noqa: F401  (AXES: this panel's)
from ._common import DATASET

EIGENSTATE_TICKS = (1, 15, 30, 45, 60)
# IBD's components are each carried by a handful of LR pairs: the top 2% (25
# rows) shows every one of them legibly, where more rows are mostly blank
TOP_LR_FRACTION = 0.02


def plot(axs):
    draw(axs, DATASET, EIGENSTATE_TICKS, TOP_LR_FRACTION)


def makeFigure():
    return common.standalone_fixed(__name__)
