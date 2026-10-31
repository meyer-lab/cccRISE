"""The IBD per-ligand NicheNet target scores, as
`figure2.figureS_nichenet.ligand_scores` draws ALAD's. Receivers are chosen
among fine cell types (`cccrise-nichenet --receiver-celltype-column
celltype`)."""

from ... import common
from ...figure2.figureS_nichenet.ligand_scores import AXES, draw  # noqa: F401
from .._common import DATASET


def plot(axs):
    draw(axs, DATASET)


def makeFigure():
    return common.standalone_fixed(__name__)
