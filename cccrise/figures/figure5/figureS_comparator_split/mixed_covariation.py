"""IBD's mixed programs' co-variation against their STAC columns."""

from ... import common
from .._recovery import program_covariation

AXES = ("split_covariation", "split_covariation")
STACKED = True  # one panel, lettered once (`common._Stack`)


def plot(axs):
    program_covariation(axs, "ibd", (1, 5))


def makeFigure():
    return common.standalone_fixed(__name__)
