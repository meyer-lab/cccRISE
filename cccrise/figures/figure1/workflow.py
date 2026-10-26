"""Placeholder for an illustration of the CCC-RISE workflow, one axes per
step: RISE, communication tensor assembly, and CPD."""

from .. import common

STEPS = (
    "RISE: multi-condition cells\n→ eigen-states",
    "Communication tensor:\nsender × receiver × LR pair × condition",
    "CPD: tensor → components",
)
AXES = ("full_third",) * len(STEPS)


def plot(axs):
    for ax, step in zip(axs, STEPS, strict=True):
        common.placeholder(ax, step)


def makeFigure():
    return common.standalone_fixed(__name__)
