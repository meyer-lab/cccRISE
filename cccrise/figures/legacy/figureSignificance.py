"""
Figure for permutation significance testing of cell-cell communication and
differential signaling in CCC-RISE.

Plots the significance tables produced by ``uv run cccrise-significance``; run
that runner first. Computes summary statistics dynamically on the fly from the
primary significance output table.

Panels, in order:

    a. Empirical P-value Distribution & Null Calibration -- density of the
       fraction of statistically significant condition x eigen-state entries
       per LR pair, showing calibration against the 5% null expectation floor.
    b. Observed vs. Permutation Null Mean -- scatter plot comparing observed
       communication score against null expectation, colored by empirical
       Z-score effect size.
    c. Differential Signaling Volcano Plot -- cross-condition communication
       difference (Group 1 - Group 0) versus -log10(p-value) across all
       sender-receiver eigen-states and LR pairs, with robust clamped outliers.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from ...runners.datasets import VALIDATION_DIR
from ...runners.provenance import sidecar_path
from ..common import getSetup, load_runner_output, subplotLabel

SELECTED_DATASET = "alad"
SIGNIFICANCE_LR_FILE = f"significance_lr_{SELECTED_DATASET}.parquet"
SIGNIFICANCE_DIFF_PARQUET = f"significance_differential_{SELECTED_DATASET}.parquet"

PALETTE_EMPHASIS = "#3a6b9c"
NULL_LINE_COLOR = "#444444"
SIG_COLOR_POS = "tab:blue"
SIG_COLOR_NEG = "tab:red"
NON_SIG_COLOR = "0.75"


def plot_pvalue_density(
    ax: plt.Axes,
    lr_df: pd.DataFrame,
    cohort_sig_rate: float | None = None,
    cohort_sig_rate_p01: float | None = None,
    title_suffix: str = "",
) -> None:
    """Panel a: Distribution of significant entry fractions across LR pairs."""
    sig_col = "FractionSignificant_p05"
    fracs = lr_df[sig_col].dropna().values

    max_frac = float(np.max(fracs)) if len(fracs) > 0 else 0.25
    x_max = max(0.25, float(np.ceil(max_frac * 20) / 20))

    sns.histplot(
        fracs,
        kde=True,
        ax=ax,
        color=PALETTE_EMPHASIS,
        bins=np.linspace(0, x_max, 26),
        stat="density",
        alpha=0.4,
        edgecolor="white",
    )

    # 5% false positive null reference
    ax.axvline(
        0.05,
        color=NULL_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Null floor (5%)",
    )

    if cohort_sig_rate is None:
        cohort_sig_rate = float(np.mean(fracs))
    if cohort_sig_rate_p01 is None and "FractionSignificant_p01" in lr_df:
        cohort_sig_rate_p01 = float(np.mean(lr_df["FractionSignificant_p01"].dropna()))

    stats_text = f"Mean Sig Rate (p<0.05): {cohort_sig_rate * 100:.1f}%"
    if cohort_sig_rate_p01 is not None:
        stats_text += f"\nMean Sig Rate (p<0.01): {cohort_sig_rate_p01 * 100:.1f}%"

    ax.text(
        0.95,
        0.85,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.8", "alpha": 0.9},
    )

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Fraction of significant entries (p < 0.05) per LR pair")
    ax.set_ylabel("Density")
    ax.set_title(f"Null Calibration {title_suffix}".strip())
    ax.legend(loc="upper right", frameon=True, fontsize=8)


def plot_observed_vs_null(
    ax: plt.Axes,
    lr_df: pd.DataFrame,
    title_suffix: str = "",
) -> None:
    """Panel b: Observed communication score vs. permutation null expectation."""
    obs = lr_df["MeanObserved"].values
    null_mean = lr_df["MeanNull"].values
    z_scores = lr_df["MeanZScore"].values

    scatter = ax.scatter(
        obs,
        null_mean,
        c=z_scores,
        cmap="coolwarm",
        s=18,
        alpha=0.7,
        edgecolors="none",
        rasterized=True,
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Mean Z-score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Identity line y = x (null expectation)
    max_val = max(np.max(obs), np.max(null_mean)) * 1.05
    min_val = min(0.0, np.min(obs), np.min(null_mean))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=NULL_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Null expectation (y = x)",
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Mean observed communication score")
    ax.set_ylabel("Mean permutation null score")
    ax.set_title(f"Observed vs. Null Mean {title_suffix}".strip())
    ax.legend(loc="lower right", frameon=True, fontsize=8)


def plot_differential_volcano(
    ax: plt.Axes,
    diff_df: pd.DataFrame | None,
    title_suffix: str = "",
) -> None:
    """Panel c: Volcano plot of cross-condition differential signaling with clamped outliers."""
    if diff_df is None or diff_df.empty:
        ax.text(
            0.5,
            0.5,
            "Differential analysis not available\n(Single-cohort or continuous group)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.5",
        )
        ax.set_title(f"Differential Communication {title_suffix}".strip())
        return

    diff_obs = diff_df["ObservedDiff"].values
    pvals = diff_df["PValue"].values
    neg_log_p = -np.log10(np.maximum(pvals, 1e-6))

    # Determine robust symmetric axis limit around 0
    abs_diff = np.abs(diff_obs)
    q_limit = float(np.quantile(abs_diff, 0.998))
    limit = max(q_limit * 1.15, 0.05)

    sig_mask = pvals < 0.05
    pos_mask = (diff_obs > 0) & sig_mask
    neg_mask = (diff_obs < 0) & sig_mask
    non_sig = ~sig_mask

    # Separate in-bounds points from clipped outliers
    outliers_pos = diff_obs > limit
    outliers_neg = diff_obs < -limit
    in_bounds = ~outliers_pos & ~outliers_neg

    ax.scatter(
        diff_obs[non_sig & in_bounds],
        neg_log_p[non_sig & in_bounds],
        c=NON_SIG_COLOR,
        s=10,
        alpha=0.4,
        edgecolors="none",
        rasterized=True,
        label="p ≥ 0.05",
    )
    ax.scatter(
        diff_obs[pos_mask & in_bounds],
        neg_log_p[pos_mask & in_bounds],
        c=SIG_COLOR_POS,
        s=14,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
        label="Elevated (p < 0.05)",
    )
    ax.scatter(
        diff_obs[neg_mask & in_bounds],
        neg_log_p[neg_mask & in_bounds],
        c=SIG_COLOR_NEG,
        s=14,
        alpha=0.8,
        edgecolors="none",
        rasterized=True,
        label="Decreased (p < 0.05)",
    )

    # Clamped outlier arrows on borders
    if np.any(outliers_pos):
        pos_out_sig = outliers_pos & sig_mask
        pos_out_nonsig = outliers_pos & ~sig_mask
        if np.any(pos_out_sig):
            ax.scatter(
                np.full(np.sum(pos_out_sig), limit),
                neg_log_p[pos_out_sig],
                c=SIG_COLOR_POS,
                marker=">",
                s=24,
                alpha=0.9,
                edgecolors="none",
                rasterized=True,
                label=f"Outliers elevated (>{limit:.2f})",
            )
        if np.any(pos_out_nonsig):
            ax.scatter(
                np.full(np.sum(pos_out_nonsig), limit),
                neg_log_p[pos_out_nonsig],
                c=NON_SIG_COLOR,
                marker=">",
                s=18,
                alpha=0.6,
                edgecolors="none",
                rasterized=True,
            )

    if np.any(outliers_neg):
        neg_out_sig = outliers_neg & sig_mask
        neg_out_nonsig = outliers_neg & ~sig_mask
        if np.any(neg_out_sig):
            ax.scatter(
                np.full(np.sum(neg_out_sig), -limit),
                neg_log_p[neg_out_sig],
                c=SIG_COLOR_NEG,
                marker="<",
                s=24,
                alpha=0.9,
                edgecolors="none",
                rasterized=True,
                label=f"Outliers decreased (<{-limit:.2f})",
            )
        if np.any(neg_out_nonsig):
            ax.scatter(
                np.full(np.sum(neg_out_nonsig), -limit),
                neg_log_p[neg_out_nonsig],
                c=NON_SIG_COLOR,
                marker="<",
                s=18,
                alpha=0.6,
                edgecolors="none",
                rasterized=True,
            )

    ax.set_xlim(-limit * 1.05, limit * 1.05)
    ax.axhline(
        -np.log10(0.05),
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="p = 0.05",
    )
    ax.set_xlabel("Mean communication difference (Group 1 - Group 0)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{Differential}\ p\mathrm{-value})$")
    ax.set_title(f"Differential Communication {title_suffix}".strip())
    ax.legend(loc="upper right", frameon=True, fontsize=7)


def makeFigure():
    """Build the 3-panel significance figure."""
    lr_df = load_runner_output(SIGNIFICANCE_LR_FILE, VALIDATION_DIR)

    diff_parquet_path = VALIDATION_DIR / SIGNIFICANCE_DIFF_PARQUET
    diff_df = pd.read_parquet(diff_parquet_path) if diff_parquet_path.exists() else None

    sidecar = sidecar_path(VALIDATION_DIR / SIGNIFICANCE_LR_FILE)
    metadata: dict = {}
    if sidecar.exists():
        with open(sidecar) as handle:
            metadata = yaml.safe_load(handle) or {}

    cohort_sig_05 = metadata.get(
        "cohort_fraction_significant_p05",
        float(lr_df["FractionSignificant_p05"].mean()),
    )
    cohort_sig_01 = metadata.get(
        "cohort_fraction_significant_p01",
        float(lr_df["FractionSignificant_p01"].mean())
        if "FractionSignificant_p01" in lr_df
        else None,
    )

    # Compute and print on-the-fly summary table
    summary_df = pd.DataFrame(
        [
            {
                "Dataset": SELECTED_DATASET,
                "TotalLRPairs": len(lr_df),
                "MeanObservedScore": float(lr_df["MeanObserved"].mean()),
                "MeanZScore": float(lr_df["MeanZScore"].mean()),
                "MedianZScore": float(lr_df["MedianZScore"].median()),
                "FractionSig_p05": float(cohort_sig_05),
                "FractionSig_p01": float(cohort_sig_01)
                if cohort_sig_01 is not None
                else np.nan,
            }
        ]
    )
    print(summary_df.to_string(index=False))

    title_suffix = f"({SELECTED_DATASET.upper()})"
    figsize = (12, 3.5)
    ax, f = getSetup(figsize, (1, 3))

    plot_pvalue_density(
        ax[0],
        lr_df,
        cohort_sig_rate=cohort_sig_05,
        cohort_sig_rate_p01=cohort_sig_01,
        title_suffix=title_suffix,
    )
    plot_observed_vs_null(ax[1], lr_df, title_suffix=title_suffix)
    plot_differential_volcano(ax[2], diff_df, title_suffix=title_suffix)

    subplotLabel(ax)
    return f
