"""Fraction of each IBD CCC-RISE population's cells with a non-zero NICHES
score minus the fraction expected from cell type, across population cutoffs
(tau).

IBD: fraction NICHES-positive minus what cell type predicts, across tau
(`_niches.tau_sweep`)."""

from ... import common
from .._niches import tau_sweep

AXES = ("mid",)


def plot(axs):
    tau_sweep(axs[0], "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
