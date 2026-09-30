"""Figure for ANOVA patient status variable associations and significance.

Runs ANOVA against multiple user-dictated patient status variables / contrasts
(e.g., across all ALAD status types: Control vs. ALAD, Control vs. Declined,
Control vs. Recovered, Declined vs. Recovered, 3-way ALAD Status, 6-month / 1-year
clinical outcomes, acute rejection, recovery trajectories, and baseline demographics)
on saved decomposition factors (e.g., ``bal_alad_factors.nc``).

Visualizes:
1. An overview comparative heatmap across all associations and CP components with
   effect sizes and FDR-adjusted p-values.
2. User-defined component weight distributions (boxplots + jittered points) across
   status categories specified via a configuration dictionary.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.figure import Figure

from ...anova import (
    _resolve_factors_and_metadata,
    correct_conditions,
    get_sample_to_group,
    prepare_group_labels,
    run_multiple_factors_anova,
)
from ...factors import ALAD_FACTORS_PATH
from ..common import subplotLabel

# Change these to switch default dataset or factor file.
SELECTED_DATASET = "alad"
FACTORS_PATH = ALAD_FACTORS_PATH

# Default association tests to run and visualize in the overview heatmap for ALAD.
# Users can modify this list or pass custom associations to makeFigure().
DEFAULT_ASSOCIATIONS: list[dict[str, Any]] = [
    # --- Primary ALAD Contrasts (Previously Tested) ---
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
    # --- Recommended Longitudinal Graft Outcomes & Rejection ---
    {
        "name": "6-Month Outcome (4-Way)",
        "group_column": "6monthcondition",
    },
    {
        "name": "6-Month CLAD/Died vs. No CLAD",
        "group_column": "6monthcondition",
        "mapping": {
            "CLAD": "CLAD/Died",
            "Died": "CLAD/Died",
            "No CLAD/Stable": "No CLAD",
            "No CLAD/Recovered": "No CLAD",
        },
        "control_label": "No CLAD",
    },
    {
        "name": "1-Year CLAD/Died vs. No CLAD",
        "group_column": "1yearcondition",
        "mapping": {
            "No CLAD/Stable": "No CLAD",
            "No CLAD/Recovered": "No CLAD",
            "CLAD/Declined": "CLAD/Died",
            "CLAD/Stable": "CLAD/Died",
            "Died": "CLAD/Died",
            "No CLAD/Declined": "No CLAD",
            "Recovered": "No CLAD",
        },
        "control_label": "No CLAD",
    },
    {
        "name": "Acute Rejection (Yes vs. No)",
        "group_column": "rejection.y",
        "levels": ["no", "yes"],
        "control_label": "no",
    },
    {
        "name": "Recovery Trajectory (5-Way)",
        "group_column": "recovered.y",
    },
    # --- Previously Tested Demographics & Baseline Covariates ---
    {
        "name": "Pre-Tx Diagnosis (ILD vs. Other)",
        "group_column": "diagnosisgroup",
        "mapping": {
            "ILD": "ILD",
            "pHTN": "Other",
            "COPD": "Other",
            "CF": "Other",
        },
        "control_label": "ILD",
    },
    {
        "name": "CMV Serostatus (IgG+ vs. IgG-)",
        "group_column": "cmvstatus2",
        "levels": ["IgG+", "IgG-"],
        "control_label": "IgG-",
    },
    {
        "name": "Sex (Male vs. Female)",
        "group_column": "sex",
        "levels": ["male", "female"],
        "control_label": "female",
    },
]

# Default user-defined box plot configurations for visualizing component weights across status categories.
# Users can modify this dictionary or pass custom boxplot definitions to makeFigure().
DEFAULT_BOXPLOTS: dict[str, dict[str, Any]] = {
    "6-Month Outcome": {
        "component": 1,
        "group_column": "6monthcondition",
        "title": "C2: 6-Month Outcome",
    },
    "Acute Rejection": {
        "component": 1,
        "group_column": "rejection.y",
        "levels": ["no", "yes"],
        "control_label": "no",
        "title": "C2: Acute Rejection",
    },
}

# Styling colors.
EMPHASIS_COLOR = "#2b5c8f"
MUTED_COLOR = "#a0aec0"
RAW_P_COLOR = "#5e3c99"
PADJ_COLOR = "#e66101"
PAIR_LINE_COLOR = "#d62728"
STRIP_JITTER = 0.2


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
                sig_mark = " *" if padj < alpha else ""
                annot_matrix[i, j] = f"{val:.3f}\np={p_str}{sig_mark}"
            else:
                annot_matrix[i, j] = "N/A"

    sns.heatmap(
        pivot_effect,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
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


def plot_category_component_weights(
    factors_adata: Any,
    component: int,
    group_column: str,
    ax: plt.Axes,
    mapping: dict[str, str] | None = None,
    levels: Sequence[str] | None = None,
    control_label: str | None = None,
    other_label: str | None = None,
    title: str | None = None,
    anova_df: pd.DataFrame | None = None,
    alpha: float = 0.05,
    patient_column: str | None = None,
) -> None:
    """Plot component loadings across patient status categories as boxplot + stripplot.

    Parameters
    ----------
    factors_adata : AnnData or factors container
        Annotated dataset with CP condition factors.
    component : int
        1-indexed CP component number to visualize.
    group_column : str
        Metadata column name in ``adata.obs``.
    ax : plt.Axes
        Matplotlib axis to plot onto.
    mapping : dict[str, str], optional
        Optional value mapping for categorical labels.
    levels : Sequence[str], optional
        Optional subset of category levels to include.
    control_label : str, optional
        Reference control category label.
    other_label : str, optional
        Combined other label.
    title : str, optional
        Subplot title.
    anova_df : pd.DataFrame, optional
        ANOVA results table for annotating test statistics in the title.
    alpha : float, default=0.05
        Significance threshold.
    patient_column : str, optional
        Metadata column identifying the patient behind each condition. Points
        of conditions sharing a patient are joined by a line.
    """
    if not hasattr(factors_adata, "obs"):
        resolved_adata, _, _ = _resolve_factors_and_metadata(factors_adata)
    else:
        resolved_adata = factors_adata

    a_mat = correct_conditions(resolved_adata)
    comp_idx = component - 1
    loadings = a_mat[:, comp_idx]

    cond_col = (
        "dsco_id"
        if "dsco_id" in resolved_adata.obs.columns
        else ("sample" if "sample" in resolved_adata.obs.columns else "patient")
    )
    raw_labels = get_sample_to_group(resolved_adata, cond_col, group_column)
    clean_labels = prepare_group_labels(
        raw_labels,
        levels=levels,
        mapping=mapping,
        control_label=control_label,
        other_label=other_label,
    )

    df_plot = pd.DataFrame(
        {
            "Condition": clean_labels.index,
            "Status": clean_labels.values,
            "Loading": loadings,
        }
    ).dropna(subset=["Status"])

    counts = df_plot["Status"].value_counts()
    categories = [cat for cat in counts.index if counts[cat] > 0]

    order_preference = [
        "No CLAD/Stable",
        "No CLAD/Recovered",
        "No CLAD",
        "CLAD",
        "CLAD/Declined",
        "CLAD/Stable",
        "CLAD/Died",
        "Died",
        "no",
        "yes",
        "control",
        "declined",
        "recovered",
        "stable",
        "improved",
        "ILD",
        "Other",
        "IgG-",
        "IgG+",
        "female",
        "male",
    ]
    categories = sorted(
        categories,
        key=lambda c: order_preference.index(c) if c in order_preference else 99,
    )

    sns.boxplot(
        data=df_plot,
        x="Status",
        y="Loading",
        order=categories,
        hue="Status",
        dodge=False,
        legend=False,
        ax=ax,
        palette="Set2",
        width=0.45,
        boxprops={"alpha": 0.75, "linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.5},
        fliersize=0,
    )
    # Jittered by hand rather than via sns.stripplot so pairing lines can be
    # drawn through the exact point positions
    rng = np.random.default_rng(42)
    positions = {cat: i for i, cat in enumerate(categories)}
    df_plot["x"] = df_plot["Status"].map(positions).astype(float) + rng.uniform(
        -STRIP_JITTER, STRIP_JITTER, len(df_plot)
    )
    ax.scatter(df_plot["x"], df_plot["Loading"], color="0.2", s=30, alpha=0.8, zorder=3)

    if patient_column is not None:
        patient_of = get_sample_to_group(resolved_adata, cond_col, patient_column)
        df_plot["Patient"] = patient_of.loc[df_plot["Condition"]].to_numpy()
        for _, samples in df_plot.groupby("Patient"):
            if len(samples) > 1:
                ax.plot(
                    samples["x"],
                    samples["Loading"],
                    color=PAIR_LINE_COLOR,
                    lw=1.2,
                    alpha=0.8,
                    zorder=2,
                )

    # Label x-ticks with sample counts and rotate if necessary
    xtick_labels = [f"{cat}\n(n={counts[cat]})" for cat in categories]
    ax.set_xticks(range(len(categories)))
    has_long_labels = any(len(str(c)) > 8 for c in categories)
    ax.set_xticklabels(
        xtick_labels,
        rotation=25 if has_long_labels else 0,
        ha="right" if has_long_labels else "center",
        fontsize=9.0 if has_long_labels else 9.5,
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"Component {component} Loading", fontsize=10.5)

    stats_str = ""
    if anova_df is not None:
        row = anova_df[anova_df["Component"] == component]
        if not row.empty:
            f_val = row["F"].iloc[0]
            padj = row["PAdj"].iloc[0]
            eta2 = row["EtaSquared"].iloc[0]
            p_str = _format_pvalue(padj)
            stats_str = (
                f"\n($F={f_val:.2f}$, $P_{{adj}}={p_str}$, $\\eta^2={eta2:.3f}$)"
            )

    ax.set_title(
        (title or f"Component {component} Weights by Status") + stats_str,
        fontsize=10.5,
        pad=6,
    )


def plot_configured_boxplots(
    factors_adata: Any,
    boxplot_configs: Sequence[dict[str, Any]] | dict[str, dict[str, Any]],
    axes: Sequence[plt.Axes] | plt.Axes,
    combined_anova_df: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> None:
    """Generate component loading box plots based on user-defined configurations.

    Parameters
    ----------
    factors_adata : Any
        Annotated dataset with CP condition factors.
    boxplot_configs : Sequence[dict[str, Any]] or dict[str, dict[str, Any]]
        User configuration dictionary or list specifying box plot parameters:
        'component', 'group_column', and optional 'mapping', 'levels',
        'control_label', 'other_label', 'title'.
    axes : Sequence[plt.Axes] or plt.Axes
        Matplotlib axis or axes to plot each box plot onto.
    combined_anova_df : pd.DataFrame, optional
        ANOVA results table for extracting test statistics to annotate titles.
    alpha : float, default=0.05
        Significance threshold.
    """
    if isinstance(boxplot_configs, dict):
        configs = [
            {"title": k, **v} if "title" not in v else {**v, "title": v.get("title", k)}
            for k, v in boxplot_configs.items()
        ]
    else:
        configs = list(boxplot_configs)

    ax_list = [axes] if isinstance(axes, plt.Axes) else list(axes)

    for cfg, ax in zip(configs, ax_list, strict=False):
        comp = int(cfg["component"])
        grp_col = str(cfg["group_column"])
        mapping = cfg.get("mapping")
        levels = cfg.get("levels")
        ctrl_label = cfg.get("control_label")
        oth_label = cfg.get("other_label")
        title = cfg.get("title")

        # Find matching ANOVA row if available
        matched_anova_df = None
        if combined_anova_df is not None:
            matching_rows = combined_anova_df[
                (combined_anova_df["GroupColumn"] == grp_col)
                | (combined_anova_df["Association"] == title)
            ]
            if not matching_rows.empty:
                matched_anova_df = matching_rows

        plot_category_component_weights(
            factors_adata=factors_adata,
            component=comp,
            group_column=grp_col,
            ax=ax,
            mapping=mapping,
            levels=levels,
            control_label=ctrl_label,
            other_label=oth_label,
            title=title,
            anova_df=matched_anova_df,
            alpha=alpha,
        )


def plot_association_effect_sizes(
    df: pd.DataFrame,
    ax: plt.Axes,
    alpha: float = 0.05,
    title: str | None = None,
    metric: str = "EtaSquared",
    highlight_component: int | None = None,
    emphasis_color: str = EMPHASIS_COLOR,
    muted_color: str = MUTED_COLOR,
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
    highlight_component : int | None
        If given, shade only this component's bar instead of the significant
        ones; significant labels stay bold either way.
    emphasis_color, muted_color : str
        Colors of the shaded and the remaining bars.
    """
    df_sorted = df.copy().sort_values("Component").reset_index(drop=True)
    components = df_sorted["Component"].astype(int).to_numpy()
    n_comp = len(components)
    positions = np.arange(n_comp)

    p_col = "PAdj" if "PAdj" in df_sorted.columns else "P"
    is_sig = (df_sorted[p_col] < alpha).to_numpy(dtype=bool)

    emphasized = (
        is_sig if highlight_component is None else (components == highlight_component)
    )
    colors = [emphasis_color if emph else muted_color for emph in emphasized]
    val_col = metric if metric in df_sorted.columns else "EtaSquared"
    values = df_sorted[val_col].to_numpy(dtype=float)

    ax.bar(
        positions,
        values,
        color=colors,
        width=0.52,
        edgecolor="0.2",
        linewidth=0.8,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{c}" for c in components], fontsize=9.5)
    ax.set_xlabel("CP Component", fontsize=10.0)
    metric_label = (
        r"Effect size ($\eta^2$)" if val_col == "EtaSquared" else "F-statistic"
    )
    ax.set_ylabel(metric_label, fontsize=10.0)

    assoc_name = (
        str(df_sorted["Association"].iloc[0])
        if "Association" in df_sorted.columns
        else ""
    )
    group_sizes = (
        df_sorted["GroupSizes"].iloc[0] if "GroupSizes" in df_sorted.columns else None
    )
    size_str = f" (N={':'.join(str(s) for s in group_sizes)})" if group_sizes else ""
    ax.set_title(title or f"{assoc_name}{size_str}", fontsize=10.5, pad=6)

    finite_vals = values[np.isfinite(values)]
    max_val = float(np.max(finite_vals)) if len(finite_vals) > 0 else 1.0
    ax.set_ylim(0, max(0.1, max_val * 1.4))
    for pos, val, padj in zip(positions, values, df_sorted[p_col], strict=True):
        if not np.isfinite(val):
            continue
        p_str = _format_pvalue(padj)
        sig_star = " *" if padj < alpha else ""
        ax.text(
            float(pos),
            val + 0.02 * max(max_val, 0.1),
            f"{val:.3f}\np={p_str}{sig_star}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="0.15",
            fontweight="bold" if padj < alpha else "normal",
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
        label=rf"$\alpha = {alpha:g}$",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"C{c}" for c in components], fontsize=9.5)
    ax.set_xlabel("CP Component", fontsize=10.0)
    ax.set_ylabel(r"$-\log_{10}(P)$", fontsize=10.0)
    ax.set_title(title or "Significance Values", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.0)


def makeFigure(
    associations: Sequence[dict[str, Any] | str] | None = None,
    boxplots: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
    dataset: str | None = None,
    factors_path: str | Path | None = None,
    global_correction: bool = False,
    alpha: float = 0.05,
) -> Figure:
    """Generate figure displaying the ANOVA summary heatmap and user-defined category box plots.

    Parameters
    ----------
    associations : Sequence[dict | str], optional
        List of patient status associations to test and plot in the heatmap.
        Defaults to ``DEFAULT_ASSOCIATIONS``.
    boxplots : Sequence[dict] or dict[str, dict], optional
        User configuration dictionary / sequence defining box plots to display.
        Defaults to ``DEFAULT_BOXPLOTS``.
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
        Figure object containing the overview heatmap and the configured category box plots.
    """
    assocs = associations if associations is not None else DEFAULT_ASSOCIATIONS
    boxplot_defs = boxplots if boxplots is not None else DEFAULT_BOXPLOTS
    target_dataset = dataset or SELECTED_DATASET
    target_factors = factors_path or FACTORS_PATH

    factors_adata, _cond_col, _grp_col = _resolve_factors_and_metadata(target_factors)

    combined_df = run_multiple_factors_anova(
        target_factors,
        associations=assocs,
        alpha=alpha,
        correction="fdr_bh",
        global_correction=global_correction,
    )

    if isinstance(boxplot_defs, dict):
        cfg_list = [
            {"title": k, **v} if "title" not in v else {**v, "title": v.get("title", k)}
            for k, v in boxplot_defs.items()
        ]
    else:
        cfg_list = list(boxplot_defs)
    n_boxplots = len(cfg_list)

    sns.set(
        style="whitegrid",
        font_scale=0.75,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    if n_boxplots == 0:
        fig_width = 7.0
        fig_height = 6.8
        f = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
        ax_heatmap = f.add_subplot(1, 1, 1)
        plot_anova_summary_heatmap(
            combined_df,
            ax_heatmap,
            alpha=alpha,
            title=f"Patient Status Associations ({target_dataset.upper()})",
        )
        subplotLabel([ax_heatmap])
        return f

    fig_width = 6.0 + 4.6 * n_boxplots
    fig_height = 6.8
    f = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    gs = gridspec.GridSpec(
        1, 1 + n_boxplots, figure=f, width_ratios=[1.35] + [1.0] * n_boxplots
    )

    ax_heatmap = f.add_subplot(gs[0, 0])
    ax_boxplots = [f.add_subplot(gs[0, 1 + i]) for i in range(n_boxplots)]

    # Panel A: Overview Heatmap
    plot_anova_summary_heatmap(
        combined_df,
        ax_heatmap,
        alpha=alpha,
        title=f"Patient Status Associations ({target_dataset.upper()})",
    )

    # Panels B, C, ...: User-defined box plots
    plot_configured_boxplots(
        factors_adata=factors_adata,
        boxplot_configs=cfg_list,
        axes=ax_boxplots,
        combined_anova_df=combined_df,
        alpha=alpha,
    )

    all_axs = [ax_heatmap, *ax_boxplots]
    subplotLabel(all_axs)
    return f
