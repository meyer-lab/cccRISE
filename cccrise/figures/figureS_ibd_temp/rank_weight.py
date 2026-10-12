"""IBD draft of the loading-by-rank figure (`figureS_rank_weight`): the same panels, reading the IBD
tables (`figureS_lr_concordance._common.DATASET`)."""

from ..figureS_lr_concordance import _common
from ..figureS_rank_weight import figureS_rank_weight


def makeFigure():
    _common.DATASET = "ibd"
    return figureS_rank_weight.makeFigure()
