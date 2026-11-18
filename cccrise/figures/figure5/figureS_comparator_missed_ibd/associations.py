"""Each PB and STAC LR column's strongest association (lowest q) with an IBD
inflammation status, beside the CCC-RISE components' associations.

IBD: `_missed.associations`."""

from ... import common
from .._missed import associations as draw

AXES = ("missed_associations",)


def plot(axs):
    draw(axs[0], "ibd")


def makeFigure():
    return common.standalone_fixed(__name__)
