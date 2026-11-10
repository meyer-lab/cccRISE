"""Figure 5C: ALAD C4's carrying pairs in the component and in its related
columns of each comparator (PB, STAC; coarse and fine apart), with each
comparator's best combination of its columns (Σ) (`_split`). ALAD C3 is in the
split supplement."""

from .. import common
from . import _split

AXES = ("fig5_columns",)


def plot(axs):
    _split.pair_heatmap(axs[0], "alad", (4,), "ALAD C4", drop=0.3)


def makeFigure():
    return common.standalone_fixed(__name__)
