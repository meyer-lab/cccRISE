"""PB's and STAC's coarse -> fine LR columns and CCC-RISE's reduced -> reported
RISE rank ones on ALAD."""

from ... import common
from .._coarse_fine import coarse_to_fine

AXES = ("labels_columns",)


def plot(axs):
    coarse_to_fine(axs[0], "alad", "ALAD")


def makeFigure():
    return common.standalone_fixed(__name__)
