"""The outcome-associated CP component's association against each LR pair's
own association in the raw communication scores."""

from .. import common, style
from . import _coordinated as C


def plot(fig):
    ax = fig.subplots()
    result = C.analysis()
    common.plot_lr_association_distribution(
        result["lr_eta"], result["component_eta"], result["top"], C.COMPONENT, ax
    )
    ax.set_title("Component vs. single LR pairs")


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["third"])
