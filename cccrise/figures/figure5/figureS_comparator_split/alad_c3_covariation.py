"""Mean pairwise rank correlation across samples of the carrying LR pairs of
ALAD CCC-RISE component 3 and of its related comparator LR columns, against
random and expression-matched null draws.

Co-variation of ALAD C3 and of its carrying columns, over all carrying
pairs against the random and the expression-matched null."""

from ... import common
from .. import _split
from ..program_covariation import KEY_TOP

AXES = ("split_columns_c3",)  # as E's heatmap


def plot(axs):
    _split.covariation(axs, "alad", (3,), key_loc="upper center", top=KEY_TOP)


def makeFigure():
    return common.standalone_fixed(__name__)
