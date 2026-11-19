"""Factor match score of each CPD component to the full-cohort fit across
leave-one-patient-out jackknife replicates, and the mean over components for
each jackknife and seed refit replicate, ALAD cohort.

Per-component CPD stability under leave-one-patient-out jackknifing, and
aggregate CPD stability under the jackknife and seed refits, on `rise_fms`'s
FMS axis (labeled there)."""

from ....validation.stability import STAGE_CPD
from ... import common
from . import _stability as S

AXES = ("fms_cpd",)


def plot(axs):
    ax = axs[0]
    S.plot_stage(ax, STAGE_CPD)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelleft=False)


def makeFigure():
    return common.standalone_fixed(__name__)
