"""Schematic of RISE projections as a common cell embedding across conditions."""

from .. import common

SCHEMATIC = True  # an illustration: no source data
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    common.placeholder(ax, "RISE projections as a\ncommon cell embedding")


def makeFigure():
    return common.standalone_fixed(__name__)
