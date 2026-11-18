"""Carrying LR pairs of each PB and STAC LR column with less than half of its
squared loading on one CCC-RISE component's carrying pairs, beside each pair's
largest loading in any CCC-RISE component, ALAD cohort.

ALAD: `_missed.contents`."""

from ... import common
from .._missed import contents as draw

AXES = ("missed_matrix",)


def plot(axs):
    draw(axs[0], "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
