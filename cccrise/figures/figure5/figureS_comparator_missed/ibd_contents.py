"""Carrying LR pairs of the IBD PB and STAC columns under half share that are
associated (q < 0.05) and carry a signalling pair CCC-RISE lacks, beside each
pair's largest loading in any CCC-RISE component (`_missed.boxed_contents`)."""

from ... import common
from .._missed import boxed_contents

AXES = ("missed_contents_ibd",)


def plot(axs):
    boxed_contents(axs[0], "ibd", key=True)


def makeFigure():
    return common.standalone_fixed(__name__)
