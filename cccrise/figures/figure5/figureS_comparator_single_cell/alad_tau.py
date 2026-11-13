"""ALAD: fraction NICHES-positive minus what cell type predicts, across tau
(`_niches.tau_sweep`)."""

from ... import common
from .._niches import tau_sweep

AXES = ("mid",)


def plot(axs):
    tau_sweep(axs[0], "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
