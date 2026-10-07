"""Association of every CP component with every ALAD outcome (one-way ANOVA
eta-squared, FDR-corrected across components within an association)."""

from ... import common
from .. import _common as S

AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    common.plot_association_heatmap(S.association_table(), ax)


def makeFigure():
    return common.standalone_fixed(__name__)
