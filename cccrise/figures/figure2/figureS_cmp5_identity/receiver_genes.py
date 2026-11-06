"""Every gene's loading on the RISE eigenstate behind the granularity
component's receiver weights (pf2_C) against its rank: the S-shaped curve,
the `N_TOP` highest-loading genes marked and named, in rank order, in columns
over the flat tail."""

import numpy as np
import pandas as pd

from ....factors import load_factors_netcdf
from ....runners.datasets import DATASETS
from ... import common, style
from ...figure3._common import plot_ranked_loadings
from .. import _common as S

EIGENSTATE = S.GRANULARITY_RECEIVER_EIGENSTATE
N_TOP = 30
LABEL_COLUMNS = 3
LABEL_X = 0.08  # first column's left edge, axes fraction
LABEL_LINE_SPACING = 1.25  # between labels, in font sizes
LABEL_COLUMN_GAP = 1  # between a column's widest label and the next, in font sizes
AXES = ("square",)


def plot(axs):
    (ax,) = axs
    factor = load_factors_netcdf(DATASETS[S.GRANULARITY_DATASET].factors_path())
    loadings = factor["pf2_C"].values[:, EIGENSTATE - 1].astype(float)
    genes = np.asarray(factor["pf2_C"]["gene"].values).astype(str)
    plot_ranked_loadings(ax, loadings, N_TOP, point_size=style.POINT_SIZE)
    ax.set_xlim(0, len(loadings) + 1)
    ax.set(xlabel="Gene rank", ylabel=f"Eigenstate {EIGENSTATE}\ngene loading")
    order = np.argsort(-loadings)
    label_columns(ax, genes[order[:N_TOP]], loadings[order[0]])
    common.source_data(
        f"eigenstate_{EIGENSTATE}_gene_loadings_ranked",
        pd.DataFrame(
            {
                "Rank": np.arange(1, len(order) + 1),
                "Gene": genes[order],
                "Loading": loadings[order],
                "Named": np.arange(len(order)) < N_TOP,
            }
        ),
        f"Every gene's loading on RISE eigenstate {EIGENSTATE} (gene factor C), "
        "the eigenstate with the largest receiver loading on CP component "
        f"{S.GRANULARITY_COMPONENT}, by rank; 'Named' marks the {N_TOP} "
        "highest-loading genes, labeled in the plot.",
    )


def label_columns(ax, names, top: float) -> None:
    """`names` in rank order down `LABEL_COLUMNS` columns, split evenly, each
    starting level with the `top` loading and just right of the previous
    column's widest label."""
    font_px = common.FACTOR_LABEL_FONT_SIZE / 72 * ax.figure.dpi
    top_y = ax.transAxes.inverted().transform(ax.transData.transform((0, top)))[1]
    per_column = -(-len(names) // LABEL_COLUMNS)
    renderer = ax.figure.canvas.get_renderer()
    x = LABEL_X
    for start in range(0, len(names), per_column):
        texts = [
            ax.text(
                x,
                top_y - row * LABEL_LINE_SPACING * font_px / ax.bbox.height,
                name,
                transform=ax.transAxes,
                va="center",
                fontsize=common.FACTOR_LABEL_FONT_SIZE,
            )
            for row, name in enumerate(names[start : start + per_column])
        ]
        width = max(t.get_window_extent(renderer).width for t in texts)
        x += (width + LABEL_COLUMN_GAP * font_px) / ax.bbox.width


def makeFigure():
    return common.standalone_fixed(__name__)
