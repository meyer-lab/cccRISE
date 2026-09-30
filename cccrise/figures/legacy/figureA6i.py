"""
Figure A6i: Per-component interpretation panel for the BAL ALAD CCC-RISE model.
Plots the ligand-receptor pairs ranked by their weight in one component, above the
distribution of that component's sender and receiver cell weights within each cell
type.

Together these are the two halves of a component's story: which cell populations
send and receive it, and which interactions carry it. The cell type x component
heatmaps (figure A6h) reduce each cell type to a single number, which cannot
distinguish a uniformly loaded population from one carrying a small high-weight
tail and reports zero for a bimodal population whose halves cancel; the LR rank
curve replaces the full LR factor heatmap with the handful of pairs that stand out
of a sparse factor.

Plots `DEFAULT_COMPONENT` by default; pass a component number to scan another,
which overwrites the same output file:

    uv run fbuild A6i        # component 1
    uv run fbuild A6i 6      # component 6
"""

import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ...utils import celltype_component_association
from ..common import get_alad_data
from ..commonFuncs.plotFactors import plot_lr_factor_rank
from ..commonFuncs.plotGeneral import celltype_grid_shape, plot_wc_histogram_grid

DEFAULT_COMPONENT = 1
CELLTYPE_COL = "broad_cell_type"
TOP_PERC = 10
# Which LR pairs get markers and labels: an int takes the top N, "tukey" takes
# those above Q3 + 1.5 * IQR, "q3" those above the 75th percentile.
LR_HIGHLIGHT: int | str = 5
# The LR factor is signed; enable to also mark the negative tail, whose largest
# magnitudes rival the positive head in some components.
LR_INCLUDE_NEGATIVE = True
PANEL_SIZE = (4.5, 3.2)
# The LR factor is sparse enough that its rank curve needs far less width than the
# histogram grid below it; the leftover width in that row is simply left empty.
LR_PANEL_SIZE = (6.0, 3.5)


def _requested_component(cp_rank: int) -> int:
    """Component from the trailing `fbuild` argument, else `DEFAULT_COMPONENT`."""
    if len(sys.argv) < 3:
        return DEFAULT_COMPONENT

    cmp = int(sys.argv[2])
    if not 1 <= cmp <= cp_rank:
        raise ValueError(f"Component {cmp} outside the 1-{cp_rank} range of this model")
    return cmp


def makeFigure():
    # Project cells rather than reading them from the factors file: cell type
    # labels come from the raw import, and the per-cell sender/receiver loadings
    # are computed from the stored RISE and CP factors.
    X = get_alad_data(project_cells=True)
    cmp = _requested_component(X.uns["A"].shape[1])
    nrows, ncols = celltype_grid_shape(X, CELLTYPE_COL)

    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )
    grid_width = PANEL_SIZE[0] * ncols
    grid_height = PANEL_SIZE[1] * nrows
    f = plt.figure(
        figsize=(grid_width, grid_height + LR_PANEL_SIZE[1]),
        layout="constrained",
    )
    grid = f.subfigures(
        2, 1, squeeze=False, height_ratios=[LR_PANEL_SIZE[1], grid_height]
    )
    top, bottom = grid[:, 0]

    lr_width = min(LR_PANEL_SIZE[0], grid_width)
    if lr_width < grid_width:
        ax_lr = top.add_subplot(
            top.add_gridspec(1, 2, width_ratios=[lr_width, grid_width - lr_width])[0, 0]
        )
    else:
        ax_lr = top.subplots()

    ranked = plot_lr_factor_rank(
        X,
        cmp,
        ax_lr,
        highlight=LR_HIGHLIGHT,
        include_negative=LR_INCLUDE_NEGATIVE,
    )

    plot_wc_histogram_grid(
        X,
        cmp,
        bottom.subplots(nrows, ncols),
        celltype_col=CELLTYPE_COL,
        top_perc=TOP_PERC,
    )
    bottom.suptitle("Cell Weight Distributions by Cell Type")

    f.suptitle(f"Component {cmp}", fontweight="bold")

    summary = pd.concat(
        {
            f"{factor_type}_{stat}": celltype_component_association(
                X,
                factor_type=factor_type,
                celltype_col=CELLTYPE_COL,
                stat=stat,
                top_perc=TOP_PERC,
            )[cmp]
            for factor_type in ("sender", "receiver")
            for stat in ("mean", "pct_top")
        },
        axis=1,
    )
    print(f"\nComponent {cmp} cell weight summary (pct_top = % in top {TOP_PERC}%):")
    print(summary.round(3).to_string())
    print(f"\nComponent {cmp} highest-weight LR pairs:")
    print(ranked.head(10).round(4).to_string(index=False))
    print(f"\nComponent {cmp} lowest-weight LR pairs:")
    print(ranked.tail(5).round(4).to_string(index=False))

    return f
