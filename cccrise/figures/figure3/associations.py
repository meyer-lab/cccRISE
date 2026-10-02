"""Association of every CP component with every ALAD outcome (one-way ANOVA
eta-squared, FDR-corrected across components within an association)."""

from .. import common, style
from . import _common as S


def plot(fig):
    ax = fig.subplots()
    common.plot_association_heatmap(S.association_table(), ax)
    ax.set_title(f"CP component - outcome associations (* FDR < {common.SIG_ALPHA})")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
