"""Loadings of IBD CCC-RISE component 1's carrying LR pairs in the component
and in the PB and STAC LR columns related to IBD C1 or C5.

IBD C1 alone, in the columns related to either C1 or C5 (the comparators
merge the two; C5's panel shows the same columns), with each comparator's
best combination (Σ) for C1."""

from ... import common
from .. import _split

AXES = ("split_columns_ibd_c1",)


def plot(axs):
    _split.pair_heatmap(
        axs[0],
        "ibd",
        (1,),
        "C1",
        drop=0.3,
        signed=True,
        slant=True,
        columns_of=(1, 5),
        colorbar=True,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
