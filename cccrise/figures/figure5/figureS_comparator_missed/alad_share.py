"""Each PB and STAC LR column's largest share of its squared loading on one
ALAD CCC-RISE component's carrying LR pairs (`_missed.share`), each fit's
block named beside it for both cohorts' panels."""

from ... import common
from .._missed import share

AXES = ("missed_bars",)


def plot(axs):
    share(axs[0], "alad", fit_names=True)


def makeFigure():
    return common.standalone_fixed(__name__)
