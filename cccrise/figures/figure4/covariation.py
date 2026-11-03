"""Figure 4B: top-pair co-variation over all pairs, the components above a stricter
null in either version."""

from .. import common
from ._strips import covariation_main

AXES = ("rect",)


def plot(axs):
    covariation_main(axs, key=True)


def makeFigure():
    return common.standalone_fixed(__name__)
