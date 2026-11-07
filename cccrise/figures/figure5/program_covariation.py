"""Figure 5E: the mixed programs' co-variation against their STAC columns."""

from .. import common
from ._recovery import program_covariation

AXES = ("fig4_covariation", "fig4_covariation")
STACKED = True  # one panel, lettered once (`common._Stack`)


def plot(axs):
    program_covariation(axs)


def makeFigure():
    return common.standalone_fixed(__name__)
