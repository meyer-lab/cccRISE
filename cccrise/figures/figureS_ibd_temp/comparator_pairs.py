"""IBD draft of the comparator-pairs figure (`figureS_comparator_pairs`): the same panels, reading the IBD
tables (`figureS_lr_concordance._common.DATASET`)."""

from ..figureS_comparator_pairs import figureS_comparator_pairs
from ..figureS_lr_concordance import _common


def makeFigure():
    _common.DATASET = "ibd"
    return figureS_comparator_pairs.makeFigure()
