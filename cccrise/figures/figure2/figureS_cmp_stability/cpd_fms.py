"""Per-component CP stability under leave-one-patient-out jackknifing,
excluding seed refits (FMS ~1.0; see Figure 2's aggregate panel)."""

from ....validation.stability import STAGE_CPD
from ... import common, style
from . import _common as S


def plot(fig, dataset: str = S.DATASET):
    S.plot_stage(fig, STAGE_CPD, dataset)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
