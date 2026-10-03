"""Cell type x RISE eigenstate association (mean projection), with the
eigenstates behind Figure 2's granularity component highlighted."""

from .. import common, style
from ..figure2._common import DATASET, RECEIVER_EIGENSTATE, SENDER_EIGENSTATE


def plot(fig):
    ax = fig.subplots()
    common.plot_celltype_eigenstate_association(
        common.get_cell_projections(DATASET),
        DATASET,
        ax,
        highlight=(SENDER_EIGENSTATE, RECEIVER_EIGENSTATE),
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 60 * style.MM))
