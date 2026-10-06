"""Cell type x RISE eigenstate association (mean projection), with the
eigenstates behind Figure 2's granularity component highlighted."""

from .. import common, style


def plot(fig):
    ax = fig.subplots()
    common.plot_celltype_eigenstate_association(
        common.get_cell_projections(common.GRANULARITY_DATASET),
        common.GRANULARITY_DATASET,
        ax,
        highlight=(
            common.GRANULARITY_SENDER_EIGENSTATE,
            common.GRANULARITY_RECEIVER_EIGENSTATE,
        ),
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 60 * style.MM))
