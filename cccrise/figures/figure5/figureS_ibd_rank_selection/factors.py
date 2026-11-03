"""The IBD RISE condition factor, as the ALAD supplement draws ALAD's
(`figure2.figureS_rise_factor_matrices.factors`)."""

from ... import common
from ...figure2.figureS_rise_factor_matrices.factors import AXES, draw  # noqa: F401
from .._common import DATASET


def plot(axs):
    draw(axs, DATASET, "terminal ileum biopsy")


def makeFigure():
    return common.standalone_fixed(__name__)
