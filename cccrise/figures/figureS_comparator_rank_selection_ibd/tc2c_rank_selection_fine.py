"""Tensor-cell2cell's elbow and near-duplicate components (fine labels)."""

from .. import common
from ._panels import DATASET, tc2c_congruence, tc2c_elbow

AXES = ("square", "square")


def plot(axs):
    tc2c_elbow(axs[:1], DATASET, "fine")
    tc2c_congruence(axs[1:], DATASET, "fine")


def makeFigure():
    return common.standalone_fixed(__name__)
