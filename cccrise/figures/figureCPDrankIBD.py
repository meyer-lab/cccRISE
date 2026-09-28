"""
Figure for IBD CPD rank selection.

Plots the cap-2000 sweep produced by
``uv run cccrise-cpdrank --dataset ibd --max-cells-per-condition 2000
--rise-rank 55``: fit R2X, the component-collinearity screen, and CORCONDIA,
using the same panels as `figureCPDrank`.
"""

from .common import getSetup, load_runner_csv, subplotLabel
from .figureCPDrank import (
    PANELS_PER_ROW,
    plot_collinearity,
    plot_core_consistency,
    plot_fit_r2x,
)

SWEEP_CSV = "cpd_rank_ibd_cap2000.csv"
LABEL = "IBD, RISE 55"


def makeFigure():
    ax, f = getSetup((4 * PANELS_PER_ROW, 3.5), (1, PANELS_PER_ROW))
    subplotLabel(ax)

    df = load_runner_csv(SWEEP_CSV)
    plot_fit_r2x(df, ax[0], f"CPD Fit $R^2X$ ({LABEL})")
    plot_collinearity(df, ax[1], f"Component Collinearity Screen ({LABEL})")
    plot_core_consistency(df, ax[2], f"Core Consistency (CORCONDIA) ({LABEL})")
    return f
