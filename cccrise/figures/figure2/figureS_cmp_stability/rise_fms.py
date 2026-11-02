"""Per-component RISE stability under leave-one-patient-out jackknifing, and
aggregate RISE stability under the jackknife and seed refits."""

from ....validation.stability import STAGE_RISE
from ... import common
from . import _common as S

# 17 components, a gap, and 2 aggregates: "wide" gives the boxes the width
# `cpd_fms`'s 8 slots get from "rect"
AXES = ("wide",)


def plot(axs):
    S.plot_stage(axs[0], STAGE_RISE)


def makeFigure():
    return common.standalone_fixed(__name__)
