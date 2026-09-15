"""
Figure A6h: Quantified cell type associations with each CCC-RISE CPD component
in the BAL ALAD data, for sender and receiver cells.

The pseudobulk decomposition (figure S5h_i) reads cell type associations straight
off its sender/receiver factors, since it aggregates cells by type before
factorizing. CCC-RISE instead resolves both cell dimensions into RISE eigenstates,
so here the per-cell sender/receiver loadings are aggregated within cell type to
put a number on the associations otherwise judged by eye from the PaCMAPs.

Each square here summarizes a whole distribution of per-cell weights; figure A6i
plots those distributions for a single component, which is what distinguishes a
uniformly loaded cell type from one carrying a small high-weight tail.
"""

import numpy as np

from ..utils import celltype_component_association
from .common import (
    get_alad_data,
    getSetup,
    subplotLabel,
)
from .commonFuncs.plotFactors import plot_celltype_association

CELLTYPE_COL = "broad_cell_type"
# See `celltype_component_association`: "mean" is the direct analogue of a
# pseudobulk factor row; "effect_size" and "enrichment" are scale-free alternatives.
STAT = "mean"
TOP_PERC = 10
NORMALIZE = "global"


def makeFigure():
    ax, f = getSetup((14, 4), (1, 2))
    subplotLabel(ax)

    # Project cells rather than reading them from the factors file: cell type
    # labels come from the raw import, and the per-cell sender/receiver loadings
    # are computed from the stored RISE and CP factors.
    X = get_alad_data(project_cells=True)

    # Scale both panels by their joint maximum so sender and receiver
    # associations are comparable to each other, not just within a panel.
    scale = max(
        float(
            np.nanmax(
                np.abs(
                    celltype_component_association(
                        X,
                        factor_type=factor_type,
                        celltype_col=CELLTYPE_COL,
                        stat=STAT,
                        top_perc=TOP_PERC,
                    ).to_numpy()
                )
            )
        )
        for factor_type in ("sender", "receiver")
    )

    for axis, factor_type, label in zip(
        ax, ["sender", "receiver"], ["Sender", "Receiver"], strict=True
    ):
        df = plot_celltype_association(
            X,
            axis,
            factor_type=factor_type,
            celltype_col=CELLTYPE_COL,
            stat=STAT,
            top_perc=TOP_PERC,
            normalize=NORMALIZE,
            scale=scale if NORMALIZE == "global" else None,
            show_counts=True,
        )
        axis.set_title(f"{label} Cell Type Association")
        print(f"\n{label} cell type association ({STAT}):")
        print(df.round(4).to_string())

    return f
