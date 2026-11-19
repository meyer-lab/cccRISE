"""Loadings of ALAD CCC-RISE component 5's carrying LR pairs in the component
and in its related PB and STAC LR columns, coarse and fine cell type labels.

Figure 5C: ALAD C5's carrying pairs in the component and in its related
columns of each comparator (PB, STAC; coarse and fine apart; C5 has only
STAC's), with a fit's best combination of its columns (Σ) where it shows two
or more (`_split`; C5's show one each, so no Σ). ALAD C3 is in the split
supplement."""

from .. import common
from . import _split

AXES = ("fig5_columns",)


def plot(axs):
    _split.pair_heatmap(axs[0], "alad", (5,), "C5", drop=0.24)


def makeFigure():
    return common.standalone_fixed(__name__)
