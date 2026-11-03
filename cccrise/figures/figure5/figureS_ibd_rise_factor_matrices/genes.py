"""The IBD RISE gene factor, as the ALAD supplement draws ALAD's
(`figure2.figureS_rise_factor_matrices.genes`), its 60 components labeled
every `TICK_STEP`."""

from ... import common
from ...figure2.figureS_rise_factor_matrices.genes import AXES, draw  # noqa: F401
from .._common import DATASET

TICK_STEP = 10


def plot(axs):
    draw(axs, DATASET)
    (ax,) = axs
    n = len(ax.get_xticks())
    ticks = [1, *range(TICK_STEP, n + 1, TICK_STEP)]
    ax.set_xticks([t - 1 for t in ticks], ticks)


def makeFigure():
    return common.standalone_fixed(__name__)
