"""Factor match score of each RISE component to the full-cohort fit across
leave-one-patient-out jackknife replicates, ALAD cohort."""

from ....validation.stability import STAGE_RISE
from ... import common
from . import _stability as S

AXES = ("fms_rise",)


def plot(axs):
    S.plot_stage(axs[0], STAGE_RISE)


def makeFigure():
    return common.standalone_fixed(__name__)
