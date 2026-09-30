"""Alignment of RISE eigenstates with annotated cell types.

Placeholder until the alignment metric is chosen. The intended source is
`utils.celltype_component_association(adata, factor_type="eigenstate",
celltype_col=DATASETS[DATASET].celltype_column, stat=...)`, a cell type x
eigenstate table, shown as a heatmap with a per-eigenstate alignment summary
and PaCMAPs of one aligned and one cross-cutting eigenstate.
"""

from .. import common, style

DATASET = "alad"


def plot(fig):
    common.placeholder(
        fig,
        "RISE eigenstate - cell type alignment\n"
        "(metric TBD: effect size / specificity / top-tail %)",
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["two_thirds"])
