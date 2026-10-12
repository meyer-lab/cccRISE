"""IBD draft of the LR-concordance figure (`figureS_lr_concordance`): the same panels, reading the IBD
tables (`figureS_lr_concordance._common.DATASET`)."""

from ..figureS_lr_concordance import _common, figureS_lr_concordance


def makeFigure():
    _common.DATASET = "ibd"
    return figureS_lr_concordance.makeFigure()
