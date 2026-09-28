"""
Figure for IBD RISE BiCV (bi-cross-validation) rank selection.

Plots the cap-2000 sweep produced by
``uv run cccrise-riserank --dataset ibd --max-cells-per-condition 2000
--ranks 30:91:5 --n-repeats 10``, marking the RISE rank carried forward.
"""

from scrise.plotting import plot_bicv_r2x

from .common import getSetup, load_runner_csv, subplotLabel

SWEEP_CSV = "rise_rank_ibd_cap2000.csv"
# One grid step below the one-SEM rule's 60; see the factor model's sidecar.
CHOSEN_RANK = 55


def makeFigure():
    ax, f = getSetup((5, 4), (1, 1))
    subplotLabel(ax)

    plot_bicv_r2x(load_runner_csv(SWEEP_CSV), ax[0])
    ax[0].axvline(CHOSEN_RANK, color="k", linestyle=":", linewidth=1)
    ax[0].set_title("BiCV rank selection (IBD, <= 2000 cells per sample)")
    return f
