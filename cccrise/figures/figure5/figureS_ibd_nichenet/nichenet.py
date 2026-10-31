"""The IBD weight-averaged NicheNet target-score percentile per CP component
against shuffled priors, as `figure2.nichenet` draws ALAD's. Receivers are
chosen among fine cell types (`cccrise-nichenet --receiver-celltype-column
celltype`)."""

from ... import common
from ...figure2.nichenet import AXES, draw  # noqa: F401
from .._common import DATASET


def plot(axs):
    draw(axs, DATASET)


def makeFigure():
    return common.standalone_fixed(__name__)
