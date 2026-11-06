"""Figure 5B: what each comparator column is."""

from .. import common
from ._recovery import census

AXES = ("fig4_census",)


def plot(axs):
    census(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
