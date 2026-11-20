"""RISE gene factor of the IBD fit for the 10% of genes with the largest scaled
absolute loading on any component, each loading divided by the component's
largest absolute loading.

The IBD RISE gene factor, as the ALAD supplement draws ALAD's
(`figure2.figureS_alad_cohort.genes`), sized as `factors` beside it."""

from ... import common
from ...figure2.figureS_alad_cohort.genes import draw
from .. import _common as S
from ._common import component_ticks

AXES = ("rect",)


def plot(axs):
    draw(axs, S.DATASET)
    component_ticks(axs[0], S.factors()["pf2_C"].shape[1])


def makeFigure():
    return common.standalone_fixed(__name__)
