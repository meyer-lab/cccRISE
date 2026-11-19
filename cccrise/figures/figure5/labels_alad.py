"""LR cos^2 between the coarse- and fine-label LR columns of PB and STAC, and
between CCC-RISE's LR columns at a reduced and at the reported RISE rank, ALAD
cohort.

Figure 5E: what PB, STAC and CCC-RISE keep and lose on ALAD when cells are
described more finely (`_coarse_fine`)."""

from .. import common
from ._coarse_fine import coarse_to_fine

AXES = ("fig5_labels",)


def plot(axs):
    coarse_to_fine(axs[0], "alad", method_names=True)


def makeFigure():
    return common.standalone_fixed(__name__)
