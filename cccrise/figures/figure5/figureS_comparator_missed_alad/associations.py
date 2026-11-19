"""Each PB and STAC LR column's strongest association (lowest q) with an ALAD
outcome, beside the CCC-RISE components' associations.

ALAD: `_missed.associations`."""

from ... import common
from .._missed import associations as draw

AXES = ("missed_associations",)


def plot(axs):
    draw(axs[0], "alad")


def makeFigure():
    return common.standalone_fixed(__name__)
