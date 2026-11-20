"""Largest congruence product between two Tensor-cell2cell components, per rank
and random start, ALAD and IBD cohorts, coarse cell type labels."""

from ... import common
from ._panels import tc2c_congruence

AXES = ("square",)


def plot(axs):
    tc2c_congruence(axs, "coarse")


def makeFigure():
    return common.standalone_fixed(__name__)
