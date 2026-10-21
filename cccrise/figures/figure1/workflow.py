"""Placeholder for an illustration of the CCC-RISE workflow."""

from .. import common

AXES = ("full",)


def plot(axs):
    (ax,) = axs
    common.placeholder(
        ax,
        "CCC-RISE workflow: RISE eigen-states → 4D communication tensor\n"
        "→ CP components (sender, receiver, LR pair, condition)",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
