"""Figure for ANOVA component associations and significance on saved factors.

Plots effect sizes (eta-squared) and significance (-log10 p-values) computed
from saved decomposition factors (e.g., ``bal_alad_factors.nc``).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..anova import run_saved_factors_anova
from ..factors import ALAD_FACTORS_PATH
from .common import getSetup, subplotLabel

# Change these to switch dataset or factor file.
SELECTED_DATASET = "alad"
FACTORS_PATH = ALAD_FACTORS_PATH

# Styling colors.
EMPHASIS_COLOR = "#3a6b9c"
MUTED_COLOR = "#8c9ba8"
RAW_P_COLOR = "#5e3c99"
PADJ_COLOR = "#e66101"


def plot_anova_effect_sizes(
    df: pd.DataFrame,
    ax: plt.Axes,
    alpha: float = 0.05,
    title: str | None = None,
    metric: str = "EtaSquared",
) -> None:
    """Plot component effect sizes (eta-squared or F-statistic) with significance annotations.

    Parameters
    ----------
    df : pd.DataFrame
        ANOVA results table containing 'Component', 'EtaSquared' (or 'F'),
        'P', and 'PAdj'.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    alpha : float, default=0.05
        Significance threshold for coloring and significance labels.
    title : str | None
        Optional title for the subplot.
    metric : str, default="EtaSquared"
        Effect size metric ('EtaSquared' or 'F').
    """
    df_sorted = df.copy().sort_values("Component").reset_index(drop=True)
    components = df_sorted["Component"].astype(int).to_numpy()
    n_comp = len(components)
    positions = np.arange(n_comp)

    p_col = "PAdj" if "PAdj" in df_sorted.columns else "P"
    is_sig = (df_sorted[p_col] < alpha).to_numpy(dtype=bool)

    colors = [EMPHASIS_COLOR if sig else MUTED_COLOR for sig in is_sig]
    val_col = metric if metric in df_sorted.columns else "EtaSquared"
    values = df_sorted[val_col].to_numpy(dtype=float)

    ax.bar(
        positions,
        values,
        color=colors,
        width=0.55,
        edgecolor="0.3",
        linewidth=0.8,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{c}" for c in components])
    ax.set_xlabel("CP Component")
    metric_label = (
        r"Effect size ($\eta^2$)" if val_col == "EtaSquared" else "F-statistic"
    )
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"Component Effect Sizes ({val_col})")

    finite_vals = values[np.isfinite(values)]
    max_val = float(np.max(finite_vals)) if len(finite_vals) > 0 else 1.0
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1.0)
    for pos, val, padj in zip(positions, values, df_sorted[p_col], strict=True):
        if not np.isfinite(val):
            continue
        stars = (
            "***"
            if padj < 0.001
            else "**"
            if padj < 0.01
            else "*"
            if padj < alpha
            else "ns"
        )
        ax.text(
            pos,
            val + 0.03 * max_val,
            f"{val:.3f}\n({stars})",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.2",
        )

    legend_handles = [
        plt.Line2D(
            [0], [0], color=EMPHASIS_COLOR, lw=6, label=f"Sig ({p_col} < {alpha:g})"
        ),
        plt.Line2D(
            [0],
            [0],
            color=MUTED_COLOR,
            lw=6,
            label=f"Not sig ({p_col} \u2265 {alpha:g})",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right")


def plot_anova_significance(
    df: pd.DataFrame,
    ax: plt.Axes,
    alpha: float = 0.05,
    title: str | None = None,
    show_padj: bool = True,
) -> None:
    """Plot -log10 raw and adjusted p-values with a significance threshold line.

    Parameters
    ----------
    df : pd.DataFrame
        ANOVA results table containing 'Component', 'P', and optionally 'PAdj'.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    alpha : float, default=0.05
        Significance threshold for reference line.
    title : str | None
        Optional title for the subplot.
    show_padj : bool, default=True
        Whether to display adjusted p-values alongside raw p-values.
    """
    df_sorted = df.copy().sort_values("Component").reset_index(drop=True)
    components = df_sorted["Component"].astype(int).to_numpy()
    n_comp = len(components)
    positions = np.arange(n_comp)

    neg_log_p = -np.log10(np.clip(df_sorted["P"].to_numpy(dtype=float), 1e-300, 1.0))
    has_padj = show_padj and "PAdj" in df_sorted.columns
    width = 0.35 if has_padj else 0.55

    if has_padj:
        neg_log_padj = -np.log10(
            np.clip(df_sorted["PAdj"].to_numpy(dtype=float), 1e-300, 1.0)
        )
        ax.bar(
            positions - width / 2,
            neg_log_p,
            width=width,
            color=RAW_P_COLOR,
            label="Raw P",
            edgecolor="0.3",
            linewidth=0.8,
        )
        ax.bar(
            positions + width / 2,
            neg_log_padj,
            width=width,
            color=PADJ_COLOR,
            label="Adjusted P (FDR)",
            edgecolor="0.3",
            linewidth=0.8,
        )
    else:
        ax.bar(
            positions,
            neg_log_p,
            width=width,
            color=RAW_P_COLOR,
            label="Raw P",
            edgecolor="0.3",
            linewidth=0.8,
        )

    threshold = -np.log10(alpha)
    max_sig = max(
        threshold,
        float(np.max(neg_log_p[np.isfinite(neg_log_p)]))
        if np.any(np.isfinite(neg_log_p))
        else 0.0,
        float(np.max(neg_log_padj[np.isfinite(neg_log_padj)]))
        if has_padj and np.any(np.isfinite(neg_log_padj))
        else 0.0,
    )
    ax.set_ylim(0, max_sig * 1.25 if max_sig > 0 else 1.0)
    ax.axhline(
        threshold,
        color="#d95f02",
        linestyle="--",
        linewidth=1.0,
        label=f"\u03b1 = {alpha:g} (-log10={threshold:.2f})",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{c}" for c in components])
    ax.set_xlabel("CP Component")
    ax.set_ylabel(r"$-\log_{10}(P)$")
    ax.set_title(title or "Significance Values")
    ax.legend(loc="upper right")


def makeFigure():
    """Generate figure with ANOVA effect sizes and significance values."""
    ax, f = getSetup((6.5, 3.2), (1, 2))
    subplotLabel(ax)

    anova_df = run_saved_factors_anova(
        FACTORS_PATH,
        control_label="control",
        other_label="alad",
        correction="fdr_bh",
    )

    plot_anova_effect_sizes(
        anova_df,
        ax[0],
        title=f"ANOVA Effect Sizes ({SELECTED_DATASET.upper()})",
    )
    plot_anova_significance(
        anova_df,
        ax[1],
        title=f"ANOVA Significance ({SELECTED_DATASET.upper()})",
    )
    return f
