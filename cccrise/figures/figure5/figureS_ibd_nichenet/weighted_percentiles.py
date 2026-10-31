"""The IBD per-ligand weight-counted target-score percentiles, as
`figure2.figureS_nichenet.weighted_percentiles` draws ALAD's. Receivers are
chosen among fine cell types (`cccrise-nichenet --receiver-celltype-column
celltype`)."""

from ... import common
from ...figure2.figureS_nichenet.weighted_percentiles import AXES, draw  # noqa: F401
from .._common import DATASET


def plot(axs):
    draw(axs, DATASET)


def makeFigure():
    return common.standalone_fixed(__name__)
