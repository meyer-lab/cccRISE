"""
Figure for RISE BiCV (bi-cross-validation) rank selection.

Plots the sweep produced by ``uv run cccrise-riserank``; run that first.

The sweep is `scrise.rank_selection.bicv`'s own long-form table, so the plot is
upstream's too: the held-out curve turning over while the in-sample fit keeps
climbing is the whole reading, and `plot_bicv_r2x` draws both.
"""

from scrise.plotting import plot_bicv_r2x

from .common import getSetup, load_runner_csv, subplotLabel

# Change these to switch dataset; must match the runner's --dataset.
SELECTED_DATASET = "alad"
SWEEP_CSV = f"rise_rank_{SELECTED_DATASET}.csv"


def makeFigure():
    ax, f = getSetup((5, 4), (1, 1))
    subplotLabel(ax)

    plot_bicv_r2x(load_runner_csv(SWEEP_CSV), ax[0])
    ax[0].set_title(f"BiCV rank selection ({SELECTED_DATASET.upper()})")
    return f
