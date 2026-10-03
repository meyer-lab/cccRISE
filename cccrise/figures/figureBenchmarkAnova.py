"""
cccRISE ANOVA benchmark: the patient-status ANOVA heatmap of `figureAnova`,
drawn once per method and comparator rank mode.

Plots `benchmark_anova_<dataset>.csv`, written by
`uv run cccrise-benchmark --dataset <dataset>`. Rows are the comparator rank
modes (each method's own rank selection, then one shared rank); cccRISE's
published model is repeated in every row as the reference column. Cell
p-values are permutation p-values, family-wise across each method's components
(`benchmark_null`), when the CSV carries them.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .common import subplotLabel
from .figureBenchmarkNull import load_runner_csv

SELECTED_DATASET = "alad"
SWEEP_CSV = f"benchmark_anova_{SELECTED_DATASET}.csv"
METHOD_ORDER = ("cccRISE", "Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
RANK_MODE_LABELS = {"own": "own rank selection", "shared": "shared rank"}


def _panel_rows(df: pd.DataFrame) -> list[list[pd.DataFrame]]:
    """One row per comparator rank mode, one panel per method, cccRISE first."""
    reference = df[df["Method"] == "cccRISE"]
    rows = []
    for mode in RANK_MODE_LABELS:
        comparators = df[df["RankMode"] == mode]
        if comparators.empty:
            continue
        rows.append(
            [reference]
            + [
                comparators[comparators["Method"] == method]
                for method in METHOD_ORDER[1:]
                if (comparators["Method"] == method).any()
            ]
        )
    return rows


def makeFigure() -> Figure:
    df = load_runner_csv(SWEEP_CSV)
    rows = _panel_rows(df)
    n_cols = max(len(row) for row in rows)
    vmax = float(df["EtaSquared"].max())
    widths = [
        max(int(row[col]["Component"].max()) if col < len(row) else 1 for row in rows)
        for col in range(n_cols)
    ]

    f, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(4.5 + 1.1 * sum(widths), 6.5 * len(rows)),
        gridspec_kw={"width_ratios": [w + 2 for w in widths]},
        layout="constrained",
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(row):
                ax.axis("off")
                continue
            panel = row[c]
            if "PermPAdj" in panel:
                panel = panel.assign(PAdj=panel["PermPAdj"])
            method, rank = panel["Method"].iloc[0], panel["Rank"].iloc[0]
            mode = (
                "published"
                if method == "cccRISE"
                else RANK_MODE_LABELS[panel["RankMode"].iloc[0]]
            )
            # One color scale across panels, so shading compares methods
            plot_anova_summary_heatmap(
                panel, ax, title=f"{method} ({mode}, rank {rank})", vmax=vmax
            )
            ax.set_xticklabels(
                [t.get_text().removeprefix("Component ") for t in ax.get_xticklabels()],
                rotation=0,
            )
            ax.set_xlabel("Component")
            if c > 0:
                ax.set_yticklabels([])

    subplotLabel([ax for ax in axes.flat if ax.axison])
    return f


# The ANOVA heatmap of the former top-level figureAnova (removed by the figure-
# suite refactor), kept here for this benchmark figure.
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
    vmax: float | None = None,
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
    vmax : float | None, optional
        Upper end of the color scale, anchored at 0, so several heatmaps can
        share one scale. Defaults to the data's own range.
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
                sig_mark = " *" if padj < alpha else ""
                annot_matrix[i, j] = f"{val:.3f}\np={p_str}{sig_mark}"
            else:
                annot_matrix[i, j] = "N/A"

    sns.heatmap(
        pivot_effect,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        vmin=None if vmax is None else 0.0,
        vmax=vmax,
        cbar_kws={
            "label": r"Effect size ($\eta^2$)" if metric == "EtaSquared" else "F",
            "shrink": 0.75,
            "pad": 0.03,
        },
        ax=ax,
        linewidths=0.8,
        linecolor="0.85",
        annot_kws={"fontsize": 8.5},
    )
    ax.set_xticklabels(
        [f"Component {c}" for c in pivot_effect.columns],
        rotation=0,
        fontsize=10.5,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        title or f"Patient Status Associations Overview ({metric})",
        fontsize=11.5,
        pad=10,
    )
