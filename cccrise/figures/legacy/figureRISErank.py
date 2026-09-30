"""
Figure for RISE BiCV (bi-cross-validation) rank selection.

Plots the sweep produced by ``uv run cccrise-riserank``; run that first.
"""

from scrise.plotting import plot_bicv_r2x

from ...runners.datasets import RANK_SELECTION_DIR
from ..common import getSetup, load_runner_output, subplotLabel

# Change these to switch dataset; must match the runner's --dataset.
SELECTED_DATASET = "alad"
SWEEP_FILE = f"rise_rank_{SELECTED_DATASET}.parquet"


def makeFigure():
    ax, f = getSetup((5, 4), (1, 1))
    subplotLabel(ax)

    plot_bicv_r2x(load_runner_output(SWEEP_FILE, RANK_SELECTION_DIR), ax[0])
    ax[0].set_title(f"BiCV rank selection ({SELECTED_DATASET.upper()})")
    return f
