"""Each PB and STAC LR column's largest share of its squared loading on one
ALAD CCC-RISE component's carrying LR pairs.

ALAD: `_missed.share`."""

from ... import common
from .._missed import share as draw

AXES = ("missed_bars",)


def plot(axs):
    draw(axs[0], "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
