"""Figure 5A: each program's best comparator column against its label ceiling."""

from .. import common
from ._recovery import program_map

AXES = ("fig4_map",)


def plot(axs):
    program_map(axs[0])


def makeFigure():
    return common.standalone_fixed(__name__)
