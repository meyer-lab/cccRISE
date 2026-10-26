"""Placeholder for an illustration of sender and receiver eigen-state
loadings combined with RISE projections into per-cell weights."""

from .. import common

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    common.placeholder(
        ax, "Eigen-state loadings × RISE\nprojections → per-cell weights"
    )


def makeFigure():
    return common.standalone_fixed(__name__)
