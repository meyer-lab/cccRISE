"""Condition loadings of the RISE component most associated with the
6-month outcome (`S.rise_association_table`), per outcome level, with a
patient's paired samples joined. The loadings are the uncorrected pf2_A the
ANOVA tested."""

from ... import common
from .. import _common as S
from ..component_weights import plot_outcome_weights

AXES = ("square",)


def plot(axs):
    plot_outcome_weights(
        axs[0],
        S.rise_conditions_adata(),
        S.rise_association_table(),
        "RISE",
        read_depth_correct=False,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
