"""Permutation significance of communication and of differential signaling,
from ``uv run cccrise-significance``: null calibration, observed against
permutation-null scores, and a volcano of the cross-group differences."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...runners.provenance import read_metadata
from .. import common, style

DATASET = "alad"
#: Differences beyond this magnitude quantile are drawn as edge arrowheads.
CLIP_QUANTILE = 0.998


def plot(fig):
    spec = DATASETS[DATASET]
    lr_table = common.load_runner_output(
        spec.significance_lr_filename(), VALIDATION_DIR
    )
    differential = pd.read_parquet(
        VALIDATION_DIR / spec.significance_differential_filename()
    )
    meta = read_metadata(VALIDATION_DIR / spec.significance_lr_filename())

    axs = fig.subplots(1, 3)
    _calibration(lr_table, meta, axs[0])
    _observed_vs_null(lr_table, axs[1])
    _volcano(differential, f"{meta['other_label']} - {meta['control_label']}", axs[2])


def _calibration(lr_table, meta, ax):
    """Fraction of significant condition x eigenstate entries per LR pair,
    against the 5% expected under the null."""
    fractions = lr_table["FractionSignificant_p05"].dropna()
    x_max = max(0.25, np.ceil(fractions.max() * 20) / 20)
    sns.histplot(
        fractions,
        kde=True,
        ax=ax,
        color=style.COLORS["emphasis"],
        bins=np.linspace(0, x_max, 26),
        stat="density",
        alpha=0.4,
        edgecolor="white",
        linewidth=style.LINE_WIDTH_THIN,
    )
    common.reference_line(ax, common.SIG_ALPHA, horizontal=False, label="Null (5%)")
    ax.text(
        0.97,
        0.75,
        f"p < 0.05: {meta['cohort_fraction_significant_p05']:.1%}\n"
        f"p < 0.01: {meta['cohort_fraction_significant_p01']:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Fraction of entries significant\nper LR pair (p < 0.05)")
    ax.set_ylabel("Density")
    ax.set_title("Null calibration")
    ax.legend(loc="upper right")


def _observed_vs_null(lr_table, ax):
    """Mean observed score against the permutation-null mean per LR pair,
    colored by mean z-score."""
    observed, null, z = (
        lr_table[c] for c in ("MeanObserved", "MeanNull", "MeanZScore")
    )
    limit = np.abs(z).max()
    points = ax.scatter(
        observed,
        null,
        c=z,
        cmap=style.CMAPS["diverging"],
        vmin=-limit,
        vmax=limit,
        s=style.POINT_SIZE_BACKGROUND,
        edgecolors="none",
        rasterized=True,
    )
    fig = ax.figure
    fig.colorbar(points, ax=ax, label="Mean z-score", shrink=0.8)
    low = min(0.0, observed.min(), null.min())
    high = max(observed.max(), null.max()) * 1.05
    ax.plot(
        [low, high],
        [low, high],
        color=style.COLORS["reference"],
        linestyle=style.LINE_STYLE_REFERENCE,
        linewidth=style.LINE_WIDTH_THIN,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Mean observed score")
    ax.set_ylabel("Mean permutation-null score")
    ax.set_title("Observed vs. null")


def _volcano(differential, contrast, ax):
    """Cross-group difference against -log10 p per sender x receiver
    eigenstate x LR pair, colored by direction where significant."""
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
    ax.set_ylim(0, neg_log_p.max() * 1.7)  # headroom for the legend
    ax.set_xlabel(f"Communication difference\n({contrast})")
    ax.set_ylabel(r"$-\log_{10}\ p$")
    ax.set_title("Differential communication")
    ax.legend(
        handles=[
            Patch(color=style.COLORS["up"], label="Up"),
            Patch(color=style.COLORS["down"], label="Down"),
            Patch(color=style.COLORS["not_significant"], label="n.s."),
        ],
        title=f"p < {common.SIG_ALPHA}",
        ncols=3,
        loc="upper center",
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
