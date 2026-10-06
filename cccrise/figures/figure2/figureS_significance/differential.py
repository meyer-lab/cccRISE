"""Differential signaling under the permutation null, from
``uv run cccrise-significance``: a volcano of the cross-group difference per
sender x receiver eigenstate x LR pair."""

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ....runners.datasets import DATASETS, VALIDATION_DIR
from ....runners.provenance import read_metadata
from ... import common, style

DATASET = "alad"
#: Differences beyond this magnitude quantile are drawn as edge arrowheads.
CLIP_QUANTILE = 0.998
AXES = ("rect",)


def plot(axs):
    (ax,) = axs
    spec = DATASETS[DATASET]
    differential = pd.read_parquet(
        VALIDATION_DIR / spec.significance_differential_filename()
    )
    meta = read_metadata(VALIDATION_DIR / spec.significance_lr_filename())
    contrast = f"{meta['other_label']} - {meta['control_label']}"

    diff = differential["ObservedDiff"].to_numpy()
    p_values = differential["PValue"].to_numpy()
    neg_log_p = -np.log10(np.maximum(p_values, 1e-6))
    colors = np.where(
        p_values < common.SIG_ALPHA,
        np.where(diff > 0, style.COLORS["up"], style.COLORS["down"]),
        style.COLORS["not_significant"],
    )
    limit = np.quantile(np.abs(diff), CLIP_QUANTILE) * 1.15
    clipped = np.abs(diff) > limit
    x = np.clip(diff, -limit, limit)
    for mask, marker, size in (
        (~clipped, "o", style.POINT_SIZE_BACKGROUND),
        (clipped & (diff > 0), ">", style.POINT_SIZE),
        (clipped & (diff < 0), "<", style.POINT_SIZE),
    ):
        ax.scatter(
            x[mask],
            neg_log_p[mask],
            c=colors[mask],
            marker=marker,
            s=size,
            edgecolors="none",
            rasterized=True,
        )
    common.reference_line(ax, -np.log10(common.SIG_ALPHA))
    ax.set_xlim(-limit * 1.05, limit * 1.05)
    ax.set_ylim(0, neg_log_p.max() * 1.05)
    ax.set_xlabel(f"Mean communication score\ndifference ({contrast})")
    ax.set_ylabel(r"$-\log_{10}\ p$")
    common.legend(
        ax,
        "right",
        handles=[
            Patch(color=style.COLORS["up"], label="Up"),
            Patch(color=style.COLORS["down"], label="Down"),
            Patch(color=style.COLORS["not_significant"], label="n.s."),
        ],
        title=f"p < {common.SIG_ALPHA}",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
