"""Per-component RISE stability under leave-one-patient-out jackknifing,
excluding seed refits (FMS ~1.0; see the aggregate panel)."""

from ....validation.stability import STAGE_RISE
from ... import common, style
from . import _common as S


def plot(fig):
    S.plot_stage(fig, STAGE_RISE)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
