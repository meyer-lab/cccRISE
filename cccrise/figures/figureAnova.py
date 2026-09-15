"""Figure for ANOVA patient status variable associations and significance.

Runs ANOVA against multiple user-dictated patient status variables / contrasts
(e.g., across all ALAD status types: Control vs. ALAD, Control vs. Declined,
Control vs. Recovered, Declined vs. Recovered, and 3-way ALAD Status) on saved
decomposition factors (e.g., ``bal_alad_factors.nc``).

Visualizes both an overview comparative heatmap across all associations and
components, alongside detailed breakdown panels for each association. Fully
generalizable to any combination of association configurations and datasets.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..anova import run_multiple_factors_anova
from ..factors import ALAD_FACTORS_PATH
from .common import getSetup, subplotLabel

# Change these to switch default dataset or factor file.
SELECTED_DATASET = "alad"
FACTORS_PATH = ALAD_FACTORS_PATH

# Default association tests to run and visualize for ALAD.
# Users can modify this list or pass custom associations to makeFigure().
DEFAULT_ASSOCIATIONS: list[dict[str, Any]] = [
    {
        "name": "Control vs. ALAD (All)",
        "group_column": "ALADstatus",
        "control_label": "control",
        "other_label": "alad",
    },
    {
        "name": "Control vs. Declined",
        "group_column": "ALADstatus",
        "levels": ["control", "declined"],
        "control_label": "control",
    },
    {
        "name": "Control vs. Recovered",
        "group_column": "ALADstatus",
        "levels": ["control", "recovered"],
        "control_label": "control",
    },
    {
        "name": "Declined vs. Recovered",
        "group_column": "ALADstatus",
        "levels": ["declined", "recovered"],
        "control_label": "declined",
    },
    {
        "name": "ALAD Status (3-Way)",
        "group_column": "ALADstatus",
    },
]

# Styling colors.
EMPHASIS_COLOR = "#3a6b9c"
MUTED_COLOR = "#8c9ba8"
RAW_P_COLOR = "#5e3c99"
PADJ_COLOR = "#e66101"


def _format_pvalue(p: float) -> str:
    """Format p-value cleanly for plot text annotations."""
    if not np.isfinite(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.1e}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def plot_anova_summary_heatmap(
    combined_df: pd.DataFrame,
    ax: plt.Axes,
    alpha: float = 0.05,
    title: str | None = None,
    metric: str = "EtaSquared",
) -> None:
    """Plot a comparative heatmap of effect sizes across all associations and components.

    Parameters
    ----------
    combined_df : pd.DataFrame
        DataFrame containing 'Association', 'Component', 'EtaSquared', and 'PAdj'.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    alpha : float, default=0.05
        Significance threshold for highlighting.
    title : str | None
        Optional title for the subplot.
    metric : str, default="EtaSquared"
        Effect size metric to display in heatmap cells ('EtaSquared' or 'F').
    """
    pivot_effect = combined_df.pivot(
        index="Association", columns="Component", values=metric
    )
    pivot_padj = combined_df.pivot(
        index="Association", columns="Component", values="PAdj"
    )

    # Maintain original association order
    unique_assocs = list(dict.fromkeys(combined_df["Association"]))
    pivot_effect = pivot_effect.reindex(unique_assocs)
    pivot_padj = pivot_padj.reindex(unique_assocs)

    # Annotate with value and p-value
    annot_matrix = np.empty(pivot_effect.shape, dtype=object)
    for i in range(pivot_effect.shape[0]):
        for j in range(pivot_effect.shape[1]):
            val = pivot_effect.iloc[i, j]
            padj = pivot_padj.iloc[i, j]
            if np.isfinite(val):
                p_str = _format_pvalue(padj)
                annot_matrix[i, j] = f"{val:.3f}\np={p_str}"
            else:
                annot_matrix[i, j] = "N/A"

    sns.heatmap(
        pivot_effect,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar_kws={
            "label": r"$\eta^2$" if metric == "EtaSquared" else "F",
            "shrink": 0.8,
        },
        ax=ax,
        linewidths=0.5,
        linecolor="0.8",
        annot_kws={"fontsize": 7.5},
    )
    ax.set_xticklabels([f"C{c}" for c in pivot_effect.columns], rotation=0)
    ax.set_xlabel("CP Component")
    ax.set_ylabel("")
    ax.set_title(title or f"Association Effect Sizes ({metric})", fontsize=9.5)


def plot_association_effect_sizes(
    df: pd.DataFrame,
    ax: plt.Axes,
    alpha: float = 0.05,
    title: str | None = None,
    metric: str = "EtaSquared",
) -> None:
    """Plot component effect sizes (eta-squared or F-statistic) for a single association.

    Parameters
    ----------
    df : pd.DataFrame
        ANOVA results table containing 'Component', 'EtaSquared' (or 'F'),
        'P', and 'PAdj'.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    alpha : float, default=0.05
        Significance threshold for coloring and annotations.
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

    assoc_name = (
        str(df_sorted["Association"].iloc[0])
        if "Association" in df_sorted.columns
        else ""
    )
    group_sizes = (
        df_sorted["GroupSizes"].iloc[0] if "GroupSizes" in df_sorted.columns else None
    )
    size_str = f" (N={':'.join(str(s) for s in group_sizes)})" if group_sizes else ""
    ax.set_title(title or f"{assoc_name}{size_str}", fontsize=9.0)

    finite_vals = values[np.isfinite(values)]
    max_val = float(np.max(finite_vals)) if len(finite_vals) > 0 else 1.0
    ax.set_ylim(0, max(0.1, max_val * 1.35))
    for pos, val, padj in zip(positions, values, df_sorted[p_col], strict=True):
        if not np.isfinite(val):
            continue
        p_str = _format_pvalue(padj)
        ax.text(
            pos,
            val + 0.02 * max(max_val, 0.1),
            f"{val:.3f}\np={p_str}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.2",
        )


def plot_association_significance(
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
        label=f"\u03b1 = {alpha:g}",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{c}" for c in components])
    ax.set_xlabel("CP Component")
    ax.set_ylabel(r"$-\log_{10}(P)$")
    ax.set_title(title or "Significance Values", fontsize=9.0)
    ax.legend(loc="upper right", fontsize=7)


def makeFigure(
    associations: Sequence[dict[str, Any] | str] | None = None,
    dataset: str | None = None,
    factors_path: str | Path | None = None,
    global_correction: bool = False,
    alpha: float = 0.05,
):
    """Generate figure running ANOVA against all specified patient status types.

    Parameters
    ----------
    associations : Sequence[dict | str], optional
        List of patient status associations to test and plot. Defaults to
        ``DEFAULT_ASSOCIATIONS`` (all ALAD status types).
    dataset : str, optional
        Dataset identifier (e.g., 'alad', 'ibd'). Defaults to ``SELECTED_DATASET``.
    factors_path : str or Path, optional
        Path to saved factors dataset. Defaults to ``FACTORS_PATH``.
    global_correction : bool, default=False
        Whether to adjust p-values globally across all associations.
    alpha : float, default=0.05
        Significance threshold.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the overview heatmap and individual association panels.
    """
    assocs = associations if associations is not None else DEFAULT_ASSOCIATIONS
    target_dataset = dataset or SELECTED_DATASET
    target_factors = factors_path or FACTORS_PATH

    combined_df = run_multiple_factors_anova(
        target_factors,
        associations=assocs,
        alpha=alpha,
        correction="fdr_bh",
        global_correction=global_correction,
    )

    unique_assocs = list(dict.fromkeys(combined_df["Association"]))
    n_assocs = len(unique_assocs)

    # 1 overview summary heatmap + n_assocs breakdown panels
    n_panels = 1 + n_assocs
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    panel_width = 3.3
    panel_height = 2.9
    ax, f = getSetup(
        (panel_width * ncols, panel_height * nrows),
        (nrows, ncols),
    )
    subplotLabel(ax)

    # Panel 0: Overview Heatmap
    plot_anova_summary_heatmap(
        combined_df,
        ax[0],
        alpha=alpha,
        title=f"Patient Status Associations ({target_dataset.upper()})",
    )

    # Panels 1..n_assocs: Individual association effect size bar plots
    for i, assoc_name in enumerate(unique_assocs):
        axis = ax[1 + i]
        assoc_df = combined_df[combined_df["Association"] == assoc_name]
        plot_association_effect_sizes(assoc_df, axis, alpha=alpha)

    # Hide any unused axes
    for extra_ax in ax[n_panels:]:
        extra_ax.set_visible(False)

    return f
