"""Per-component CP stability under resampling of cells, conditions, and LR
pairs, against the seed-only reproducibility ceiling."""

from ....validation.component_stability import STAGE_CPD
from ... import common, style
from . import _common as S


def plot(fig):
    S.plot_stage(fig, STAGE_CPD)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"], legend=S.legend)
