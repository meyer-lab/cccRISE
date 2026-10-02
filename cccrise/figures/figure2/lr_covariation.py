"""Co-variation across conditions of the outcome-associated CP component's top
LR pairs, against random, expression-matched, and other associated LR pairs."""

from .. import common, style
from . import _coordinated as C


def plot(fig):
    ax = fig.subplots()
    result = C.analysis()
    common.plot_lr_covariation(
        result["observed"], result["baselines"], C.COMPONENT, C.N_TOP, ax
    )
    ax.set_title("Co-variation of LR pairs")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
