"""Shiny app for interactively exploring RISE / CCC-RISE decomposition output.

Run with:
    shiny run --reload cccrise/app/app.py
    shiny run --host 0.0.0.0 --port 8000 --reload cccrise/app/app.py
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.rcsetup import cycler
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from scrise.plotting import plot_condition_factors as plot_rise_condition_factors
from scrise.plotting import plot_eigenstate_factors as plot_rise_eigenstate_factors
from scrise.plotting import plot_gene_factors as plot_rise_gene_factors
from shiny import App, reactive, render, ui

from cccrise.anova import run_saved_factors_anova
from cccrise.app import colors
from cccrise.app.data import (
    DATASETS,
    LoadedDataset,
    load_dataset,
    load_rank_selection,
    load_raw_dataset,
    load_rise_factors,
)
from cccrise.app.lr_pairs import parse_lr_pairs
from cccrise.app.plots import (
    celltype_grid_shape,
    enlarge_points,
    pacmap_axes,
    plot_categorical_scatter,
    plot_celltype_association,
    plot_celltype_points,
    plot_cp_factor_heatmaps,
    plot_interaction_heatmap,
    plot_lr_rank,
    plot_value_pacmap,
    plot_weight_boxplot,
    plot_weight_histogram_grid,
)
from cccrise.figures import style
from cccrise.figures.common import (
    plot_association_effect_sizes,
    plot_category_component_weights,
    plot_collinearity_medians,
    plot_cpd_fit_r2x,
    plot_cpd_imputation_r2x,
    plot_rise_rank,
)
from cccrise.figures.figure3._common import ASSOCIATIONS
from cccrise.figures.figureS_alad_pacmaps.weight_pacmaps import SATURATION_PERCENTILE
from cccrise.runners.datasets import DATASETS as RUNNER_DATASETS
from cccrise.utils import celltype_component_association

DATASET_CHOICES = {key: spec.label for key, spec in DATASETS.items()}

LR_HIGHLIGHT_CHOICES = {
    "5": "Top 5 Pairs",
    "10": "Top 10 Pairs",
    "15": "Top 15 Pairs",
    "tukey": "Tukey Outliers (Q3 + 1.5*IQR)",
    "q3": "Q3 (75th Percentile)",
}

# Statistics offered by `celltype_component_association`.
ASSOC_STAT_CHOICES = {
    "mean": "Mean Cell Loading",
    "effect_size": "Cohen's d vs Other Cells",
    "enrichment": "log2 Enrichment",
    "pct_top": "% in Top Percentile",
}
# Association stats that depend on the Top Cell % Threshold.
TOP_PERC_STATS = ("enrichment", "pct_top")

# Multipliers applied to every plot's text by `scale_fonts`, relative to the
# sizes the shared plotting functions set for the paper figures.
TITLE_SCALE = 1.3
AXIS_LABEL_SCALE = 1.3
TICK_LABEL_SCALE = 1.15
CBAR_TICK_LABEL_SCALE = 1.3
TEXT_SCALE = 1.15

# Section headers, 1.5x Bootstrap's h4 (1.5rem) and h6 (1rem).
HEADER_CSS = "h4 { font-size: 2.25rem; } h6 { font-size: 1.5rem; }"

# "Neither" is usually the majority category, so it's muted rather than drawn
# from the same accented palette as the other three.
STRENGTH_CATEGORY_COLORS = {
    "Both": colors.PINK,
    "Neither": colors.LIGHT_GREY,
    "Receiver Only": colors.RECEIVER,
    "Sender Only": colors.SENDER,
}

# Sender/receiver scatter component selects (x and y axes).
SCATTER_COMPONENT_INPUTS = {
    "scatter_x_cmp": "Component X (x axis)",
    "scatter_y_cmp": "Component Y (y axis)",
}
ROLE_WEIGHT_KEYS = {"Sender": "sc_B", "Receiver": "rc_C"}
# Scatter output size (px): one panel when X == Y, else the 2x2 role grid.
SCATTER_SINGLE_PX = (900, 700)
SCATTER_GRID_PX = (1400, 1150)

# Plots that don't set their own colors draw from the shared palette.
plt.rcParams["axes.prop_cycle"] = cycler(color=colors.categorical_colors(8))

# `figures/common.py` sizes legend handles for 6 pt print figures, and handles
# are fixed at legend creation (`scale_fonts` only enlarges the text), so build
# legends at on-screen size with square color boxes instead.
plt.rcParams.update(
    {
        "legend.fontsize": plt.rcParams["xtick.labelsize"],
        "legend.handlelength": 1.0,
        "legend.handleheight": 1.0,
        "legend.markerscale": 1.0,
    }
)

# ALAD: Figure 3's curated comparisons, reused as-is (single source of
# truth) rather than re-hardcoded here.
ALAD_COMPARISONS: dict[str, dict] = {cfg["name"]: cfg for cfg in ASSOCIATIONS}

# ALAD's default comparison tests every level of the dataset spec's outcome
# column, matching the stability runner's ANOVA default.
_REGROUPING_KEYS = {"mapping", "levels", "control_label", "other_label"}
ALAD_DEFAULT_COMPARISON = next(
    name
    for name, cfg in ALAD_COMPARISONS.items()
    if cfg["group_column"] == RUNNER_DATASETS["alad"].outcome_column
    and not _REGROUPING_KEYS & cfg.keys()
)

DATASET_COMPARISONS: dict[str, dict[str, dict]] = {"alad": ALAD_COMPARISONS}
DEFAULT_COMPARISONS: dict[str, str] = {"alad": ALAD_DEFAULT_COMPARISON}


def scrollable_plot(output_id: str, width_px: int, height_px: int) -> ui.Tag:
    """A plot output pinned to a fixed pixel size, in a horizontally
    scrollable wrapper.

    `ui.output_plot` defaults to a responsive (percentage) width, which
    combined with a fixed pixel height stretches or squashes a
    fixed-aspect-ratio matplotlib figure as its container narrows (e.g. the
    sidebar being toggled) -- heatmap rows compress until their labels
    overlap, or the whole panel shrinks past legibility. Fixing both
    dimensions keeps every panel at a legible, undistorted size regardless
    of viewport width; a narrower window scrolls instead of squishing.
    """
    return ui.div(
        ui.output_plot(output_id, width=f"{width_px}px", height=f"{height_px}px"),
        style="overflow-x: auto; max-width: 100%;",
    )


def scale_fonts(fig: Figure, scale_ticks: bool = True) -> None:
    """Enlarge a figure's text in place, by text category (see `*_SCALE`).
    `scale_ticks=False` leaves non-colorbar tick labels at their size.
    Legends are then set to the figure's tick label size."""

    def grow(text: Text, factor: float) -> None:
        text.set_fontsize(float(text.get_fontsize()) * factor)

    for text in fig.texts:
        grow(text, TEXT_SCALE)
    if (suptitle := getattr(fig, "_suptitle", None)) is not None:
        grow(suptitle, TITLE_SCALE)

    # Inset axes (e.g. a colorbar via `ax.inset_axes`) aren't in `fig.axes`.
    all_axes = [*fig.axes, *(child for ax in fig.axes for child in ax.child_axes)]
    for ax in all_axes:
        grow(ax.title, TITLE_SCALE)
        grow(ax.xaxis.label, AXIS_LABEL_SCALE)
        grow(ax.yaxis.label, AXIS_LABEL_SCALE)
        for text in ax.texts:
            grow(text, TEXT_SCALE)

        if hasattr(ax, "_colorbar"):
            tick_scale = CBAR_TICK_LABEL_SCALE
        else:
            tick_scale = TICK_LABEL_SCALE if scale_ticks else 1.0
        for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            labels = axis.get_ticklabels()
            if labels:
                ax.tick_params(
                    axis=axis_name,
                    labelsize=float(labels[0].get_fontsize()) * tick_scale,
                )

    legend_size = figure_tick_size(fig)
    legends = [*fig.legends, *(ax.get_legend() for ax in all_axes)]
    for legend in filter(None, legends):
        for text in [*legend.get_texts(), legend.get_title()]:
            text.set_fontsize(legend_size)


def figure_tick_size(fig: Figure) -> float:
    """Size of the first visible non-colorbar tick label, else the scaled
    default (PaCMAPs hide their ticks)."""
    for ax in fig.axes:
        if hasattr(ax, "_colorbar") or not ax.get_visible():
            continue
        for axis in (ax.xaxis, ax.yaxis):
            if labels := axis.get_ticklabels():
                return float(labels[0].get_fontsize())
    default = FontProperties(size=plt.rcParams["xtick.labelsize"])
    return default.get_size_in_points() * TICK_LABEL_SCALE


def place_legends(fig: Figure) -> None:
    """Move in-axes legends to matplotlib's least-overlapping spot ("best").
    Legends already anchored elsewhere (e.g. beside a heatmap or a per-cell
    scatter) are left alone."""
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None or getattr(legend, "_bbox_to_anchor", None) is not None:
            continue
        legend.set_loc("best")


def styled_figure(scale_ticks: bool = True):
    """Decorator applying `scale_fonts` and `place_legends` to the figure a
    render function returns."""

    def decorator(render_fn):
        @functools.wraps(render_fn)
        def wrapper():
            fig = render_fn()
            if fig is not None:
                scale_fonts(fig, scale_ticks=scale_ticks)
                place_legends(fig)
            return fig

        return wrapper

    return decorator


def mark_selected_rank(ax: plt.Axes, rank: int) -> None:
    """Dashed vertical line at the rank the loaded factors were fit at."""
    ax.axvline(
        rank, color="black", ls="--", lw=1.2, zorder=0, label=f"Loaded rank ({rank})"
    )


def plot_celltype_pacmap(ds: LoadedDataset, ax: plt.Axes) -> None:
    plot_celltype_points(ax, ds.adata, ds.spec.key, np.asarray(ds.adata.obsm["PaCMAP"]))
    pacmap_axes(ax)
    ax.set_title("Cell Type")


def top_perc_slider(input_id: str) -> ui.Tag:
    return ui.input_slider(
        input_id, "Top Cell % Threshold", min=1, max=50, value=10, step=1
    )


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("dataset", "Dataset", choices=DATASET_CHOICES),
        ui.input_slider("component", "CP Component", min=1, max=1, value=1, step=1),
        ui.output_ui("projections_notice"),
        width=300,
    ),
    ui.tags.style(HEADER_CSS),
    ui.navset_tab(
        ui.nav_panel(
            "Rank Selection",
            ui.output_ui("rank_selection_body"),
        ),
        ui.nav_panel(
            "RISE Factors",
            ui.markdown(
                "Heatmaps of the RISE condition (A), eigenstate (B), and "
                "gene (C) factor matrices. Condition factors are "
                "standardized per component. The C panel shows only the "
                "genes with the largest max |weight| across components."
            ),
            ui.input_slider(
                "n_genes", "Genes shown in C panel", min=10, max=200, value=50, step=5
            ),
            scrollable_plot("rise_factor_heatmaps", 1400, 800),
        ),
        ui.nav_panel(
            "CP Factors",
            ui.markdown(
                "Heatmaps of the condition (A), sender (B), receiver (C), "
                "and ligand-receptor (D) factor matrices. The trim "
                "threshold slider drops low-weight LR pairs from the D "
                "panel."
            ),
            ui.input_slider(
                "lr_weight",
                "LR factor trim threshold (min |weight|)",
                min=0,
                max=0.3,
                value=0.06,
                step=0.01,
            ),
            scrollable_plot("factor_heatmaps", 1400, 650),
        ),
        ui.nav_panel(
            "Component Explorer",
            ui.output_ui("explorer_body"),
        ),
        ui.nav_panel(
            "Cell Projections",
            ui.output_ui("projections_body"),
        ),
        ui.nav_panel(
            "Summary Tables",
            ui.output_ui("summary_tables_body"),
        ),
    ),
    title="CCC-RISE Explorer",
)


def server(input, output, session):
    @reactive.calc
    def dataset():
        return load_dataset(input.dataset())

    @reactive.calc
    def cp_rank():
        return int(dataset().adata.uns["A"].shape[1])

    @reactive.effect
    @reactive.event(cp_rank)
    def _sync_component_slider():
        # Gated on cp_rank (only changes when the dataset does) rather than
        # running on every component-slider tick: re-issuing update_slider
        # on each tick raced with the slider's own client-side updates while
        # dragging, producing the reported jumping between low and high
        # values near the top of the range.
        rank = cp_rank()
        with reactive.isolate():
            current = input.component()
        ui.update_slider(
            "component",
            max=rank,
            value=min(current, rank) if current else 1,
        )

    @render.ui
    def projections_notice():
        if not dataset().has_projections:
            return ui.markdown(
                "⚠️ This dataset's raw single-cell data was not found "
                "locally, so per-cell projections (recomputed from it) are "
                "unavailable; only factor matrices are shown."
            )
        return None

    # ------------------------------------------------------------------
    # Rank Selection
    # ------------------------------------------------------------------
    @reactive.calc
    def rise_rank():
        return int(dataset().adata.uns["B"].shape[0])

    @render.ui
    def rank_selection_body():
        sweeps = load_rank_selection(input.dataset())
        if sweeps.rise is None and sweeps.cpd is None:
            return ui.markdown(
                "No cached rank-selection sweeps for this dataset. Produce them "
                "with `uv run cccrise-riserank` and `uv run cccrise-cpdrank`."
            )
        notes = [
            (
                "Cached rank-selection sweeps (as in manuscript Figure 2a-c): RISE "
                "full-data fit R2X and bi-cross-validation (BiCV) train and test "
                "R2X, CPD fit and k-fold imputation R2X, and CPD component "
                "collinearity. Dashed lines mark the RISE rank "
                f"(**{rise_rank()}**) and CP rank (**{cp_rank()}**) of the loaded "
                "factors used by the remaining tabs."
            )
        ]
        if sweeps.cpd_rise_rank is not None and sweeps.cpd_rise_rank != rise_rank():
            notes.append(
                f"⚠️ The CPD sweep was run on an interaction tensor built at "
                f"RISE rank {sweeps.cpd_rise_rank}, not the loaded factors' "
                f"RISE rank {rise_rank()}."
            )
        return ui.TagList(
            ui.markdown("\n\n".join(notes)),
            scrollable_plot("rank_selection_plot", 1800, 550),
        )

    @render.plot
    @styled_figure()
    def rank_selection_plot():
        sweeps = load_rank_selection(input.dataset())
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="tight")

        if sweeps.rise is None:
            axes[0].axis("off")
            axes[0].text(0.5, 0.5, "No cached RISE BiCV sweep", ha="center")
        else:
            plot_rise_rank(sweeps.rise, axes[0])
            axes[0].set_title("RISE BiCV Rank Selection")
            mark_selected_rank(axes[0], rise_rank())

        if sweeps.cpd is None:
            for ax in axes[1:]:
                ax.axis("off")
                ax.text(0.5, 0.5, "No cached CPD sweep", ha="center")
        else:
            plot_cpd_fit_r2x(
                sweeps.cpd, axes[1], color=style.COLORS["fit_full"], marker=None
            )
            if sweeps.cpd_imputation is not None:
                plot_cpd_imputation_r2x(sweeps.cpd_imputation, axes[1])
            axes[1].set_title("CPD Fit and Imputation $R^2X$")
            plot_collinearity_medians(axes[2], input.dataset())
            axes[2].set_title("CPD Collinearity by Mode (10 Seeded Starts)")
            for ax in axes[1:]:
                mark_selected_rank(ax, cp_rank())
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if ax.get_legend_handles_labels()[0]:
                # Long enough handles for the dashed selected-rank line to read
                ax.legend(handlelength=2)
        return fig

    # ------------------------------------------------------------------
    # RISE Factors
    # ------------------------------------------------------------------
    @render.plot
    @styled_figure(scale_ticks=False)
    def rise_factor_heatmaps():
        ds = dataset()
        X = load_rise_factors(input.dataset())
        fig, axes = plt.subplots(1, 3, figsize=(20, 8), layout="tight")

        # log10 is undefined for the negative condition weights some fits have
        plot_rise_condition_factors(
            X, axes[0], cond=ds.spec.condition_col, log_transform=False
        )
        axes[0].set_title("Condition Factor (A)")

        plot_rise_eigenstate_factors(X, axes[1])
        axes[1].set_title("Eigenstate Factor (B)")

        # Trim threshold that keeps the top `n_genes` genes by max |weight|
        max_weight = np.sort(np.abs(np.asarray(X.varm["Pf2_C"])).max(axis=1))[::-1]
        threshold = max_weight[min(input.n_genes(), max_weight.size - 1)]
        plot_rise_gene_factors(X, axes[2], weight=threshold)
        axes[2].set_title("Gene Factor (C)")

        return fig

    # ------------------------------------------------------------------
    # Factor Matrices
    # ------------------------------------------------------------------
    @render.plot
    @styled_figure(scale_ticks=False)
    def factor_heatmaps():
        ds = dataset()
        X = ds.adata
        fig, axes = plt.subplots(1, 4, figsize=(20, 8), layout="tight")
        plot_cp_factor_heatmaps(X, axes, ds.spec.condition_col, input.lr_weight())
        for ax, title in zip(
            axes,
            (
                "Condition Factor (A)",
                "Sender Eigenstate Factor (B)",
                "Receiver Eigenstate Factor (C)",
                "Ligand-Receptor Factor (D)",
            ),
            strict=True,
        ):
            ax.set_title(title)

        return fig

    # ------------------------------------------------------------------
    # Component Explorer (top LR pairs, per-cell-type distributions, and
    # the single-pair interaction heatmap, all keyed on the sidebar's
    # dataset/component selections)
    # ------------------------------------------------------------------
    @reactive.calc
    def simple_lr_pairs():
        ds = dataset()
        pairs = parse_lr_pairs(np.asarray(ds.adata.uns["lr_pairs"]))
        return [p for p in pairs if p.is_simple_receptor]

    @reactive.calc
    def patient_status_result():
        ds = dataset()
        comparisons = DATASET_COMPARISONS.get(ds.spec.key, {})
        name = input.patient_status_comparison()
        cfg = comparisons.get(name)
        if cfg is None:
            return None, None, "no comparison selected."
        raw = load_raw_dataset(input.dataset())
        if raw.adata is None:
            return None, None, raw.error or "raw dataset unavailable."
        try:
            df = run_saved_factors_anova(
                factors=raw.adata,
                condition_column=ds.spec.condition_col,
                group_column=cfg["group_column"],
                control_label=cfg.get("control_label"),
                other_label=cfg.get("other_label"),
                levels=cfg.get("levels"),
                mapping=cfg.get("mapping"),
                correction="fdr_bh",
                association_name=name,
            )
        except ValueError as exc:
            return None, None, str(exc)
        return df, cfg, None

    @render.ui
    def patient_status_ui():
        _, _, error = patient_status_result()
        if error is not None:
            return ui.markdown(f"_Patient status comparison unavailable: {error}_")
        return scrollable_plot("patient_status_plot", 1200, 500)

    @render.plot
    @styled_figure()
    def patient_status_plot():
        df, cfg, error = patient_status_result()
        if error is not None:
            return None
        raw = load_raw_dataset(input.dataset())
        if raw.adata is None:
            return None
        cmp = input.component()
        name = input.patient_status_comparison()
        runner_spec = RUNNER_DATASETS.get(input.dataset())
        patient_column = runner_spec.patient_column if runner_spec else None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="tight")
        plot_association_effect_sizes(
            df,
            axes[0],
            highlight_component=cmp,
            emphasis_color=colors.HIGHLIGHT,
            muted_color=colors.MUTED,
        )
        plot_category_component_weights(
            raw.adata,
            component=cmp,
            group_column=cfg["group_column"],
            mapping=cfg.get("mapping"),
            levels=cfg.get("levels"),
            control_label=cfg.get("control_label"),
            other_label=cfg.get("other_label"),
            xlabel="",
            ax=axes[1],
            anova_df=df,
            patient_column=patient_column,
            condition_column=DATASETS[input.dataset()].condition_col,
            palette=colors.categorical_colors(8),
        )
        axes[1].set_title(name)
        return fig

    @render.ui
    def explorer_body():
        ds = dataset()
        if not ds.has_projections:
            return ui.markdown(
                "Per-cell weights are unavailable for this dataset "
                "(raw single-cell data or projections not found)."
            )

        comparisons = DATASET_COMPARISONS.get(ds.spec.key, {})
        sections = [
            ui.h4("Condition Associations"),
            ui.markdown(
                "ANOVA effect sizes and a per-component boxplot for a "
                "chosen patient status comparison, computed from the raw "
                "single-cell data. Comparison selects the patient status "
                "variable/grouping tested; both panels use the component "
                "selected in the sidebar."
            ),
            ui.input_select(
                "patient_status_comparison",
                "Comparison",
                choices=list(comparisons),
                selected=DEFAULT_COMPARISONS.get(ds.spec.key),
            ),
            ui.output_ui("patient_status_ui"),
            ui.hr(),
        ]

        sections.extend(
            [
                ui.h4("Ligand-Receptor Pairs"),
                ui.markdown(
                    "Every LR pair's weight for the selected component, ranked "
                    "from highest to lowest. Highlight Rule sets which pairs "
                    "get markers and labels; Include Negative LR Tail also "
                    "marks the low end; Max Callout Labels caps labels per "
                    "side."
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.input_select(
                            "lr_highlight_mode",
                            "Highlight Rule",
                            choices=LR_HIGHLIGHT_CHOICES,
                            selected="5",
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_checkbox(
                            "lr_include_negative",
                            "Include Negative LR Tail",
                            value=False,
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_numeric(
                            "lr_max_labels",
                            "Max Callout Labels",
                            value=10,
                            min=1,
                            max=30,
                        ),
                    ),
                ),
                scrollable_plot("lr_rank_curve_plot", 1100, 450),
                ui.hr(),
                ui.h4("Sender/Receiver Cells"),
                ui.markdown(
                    "Heatmap of every cell type's association with every "
                    "component, for sender and receiver. Association Heatmap "
                    "Stat selects the statistic shown; Top Cell % "
                    "Threshold sets the percentile used by the "
                    "enrichment and % Top statistics."
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.input_select(
                            "assoc_stat",
                            "Association Heatmap Stat",
                            choices=ASSOC_STAT_CHOICES,
                            selected="mean",
                        ),
                    ),
                    ui.column(4, top_perc_slider("assoc_top_perc")),
                ),
                scrollable_plot("assoc_heatmap_plot", 1300, 500),
                ui.hr(),
                ui.h4("LR Pair Interactions"),
            ]
        )

        raw = load_raw_dataset(input.dataset())
        if raw.error is not None or raw.adata is None:
            sections.append(ui.markdown(f"⚠️ {raw.error or 'Raw dataset unavailable.'}"))
            return ui.TagList(*sections)

        celltypes = sorted(raw.adata.obs[ds.spec.celltype_col].unique().tolist())
        pair_labels = [p.label for p in simple_lr_pairs()]

        sections.extend(
            [
                ui.markdown(
                    "Binned expression-product heatmap for one "
                    "ligand-receptor pair. Ligand-Receptor Pair selects the "
                    "pair; Sender/Receiver Cell Type select the two cell "
                    "populations."
                ),
                ui.row(
                    ui.column(
                        4,
                        ui.input_selectize(
                            "lr_pair", "Ligand-Receptor Pair", choices=pair_labels
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_select(
                            "sender_celltype", "Sender Cell Type", choices=celltypes
                        ),
                    ),
                    ui.column(
                        4,
                        ui.input_select(
                            "receiver_celltype", "Receiver Cell Type", choices=celltypes
                        ),
                    ),
                ),
                scrollable_plot("lr_interaction_plot", 700, 500),
            ]
        )
        return ui.TagList(*sections)

    @render.plot
    @styled_figure()
    def lr_rank_curve_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        highlight_raw = input.lr_highlight_mode()
        highlight: int | str = (
            int(highlight_raw) if highlight_raw.isdigit() else highlight_raw
        )
        include_neg = input.lr_include_negative()
        max_labels = int(input.lr_max_labels() or 10)

        fig, ax = plt.subplots(figsize=(10, 4.5), layout="tight")
        plot_lr_rank(
            X,
            cmp,
            ax,
            highlight=highlight,
            include_negative=include_neg,
            max_labels=max_labels,
            colors=(colors.ORANGE, colors.BLUE),
        )
        return fig

    @render.plot
    @styled_figure()
    def distributions_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        view_mode = input.dist_view_mode()
        top_perc = float(input.dist_top_perc())

        if view_mode == "hist":
            nrows, ncols = celltype_grid_shape(X, celltype_col)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), layout="tight"
            )
            plot_weight_histogram_grid(
                X,
                cmp,
                axes,
                celltype_col=celltype_col,
                top_perc=top_perc,
                log_y=input.dist_log_y(),
                colors=(colors.SENDER, colors.RECEIVER),
            )
            return fig

        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="tight")
            for ax, role, color in zip(
                axes,
                ("sender", "receiver"),
                (colors.SENDER, colors.RECEIVER),
                strict=True,
            ):
                plot_weight_boxplot(X, cmp, ax, celltype_col, role, color)
            return fig

    @render.plot
    @styled_figure()
    def assoc_heatmap_plot():
        ds = dataset()
        X = ds.adata
        celltype_col = ds.spec.celltype_col
        stat = input.assoc_stat()
        top_perc = float(input.assoc_top_perc())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="tight")
        scale = max(
            float(
                np.nanmax(
                    np.abs(
                        celltype_component_association(
                            X,
                            factor_type=ft,
                            celltype_col=celltype_col,
                            stat=stat,
                            top_perc=top_perc,
                        ).to_numpy()
                    )
                )
            )
            for ft in ("sender", "receiver")
        )

        for ax, ft, label in zip(
            axes, ["sender", "receiver"], ["Sender", "Receiver"], strict=True
        ):
            plot_celltype_association(
                X,
                ax,
                factor_type=ft,
                celltype_col=celltype_col,
                stat=stat,
                top_perc=top_perc,
                scale=scale,
                cbar=ft == "receiver",
            )
            ax.set_title(f"{label} Cell Type Association")
        # Both panels share rows and a color scale: label rows once, on the left
        axes[1].tick_params(axis="y", labelleft=False)
        return fig

    @render.plot
    @styled_figure()
    def lr_interaction_plot():
        ds = dataset()
        raw = load_raw_dataset(input.dataset())
        if raw.adata is None:
            return None
        X = raw.adata
        cmp = input.component()

        pair_map = {p.label: p for p in simple_lr_pairs()}
        pair = pair_map.get(input.lr_pair())
        if pair is None:
            return None

        ligand_arg = pair.ligand
        receptor_arg = pair.receptor_subunits[0]

        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            plot_interaction_heatmap(
                X,
                cmp=cmp,
                ligand=ligand_arg,
                receptor=receptor_arg,
                ax=ax,
                sender_celltype=input.sender_celltype(),
                receiver_celltype=input.receiver_celltype(),
                celltype_col=ds.spec.celltype_col,
            )
        except (KeyError, AssertionError) as exc:
            ax.axis("off")
            ax.text(
                0.5, 0.5, f"Unable to plot: {exc}", ha="center", va="center", wrap=True
            )
        return fig

    # ------------------------------------------------------------------
    # Cell Projections
    # ------------------------------------------------------------------
    @render.ui
    def projections_body():
        if not dataset().has_pacmap:
            return ui.markdown(
                "Per-cell PaCMAP coordinates are unavailable: they need "
                "this dataset's raw single-cell data locally, and a factors "
                "file from `uv run cccrise-run` without `--no-embedding`."
            )
        celltypes = sorted(
            dataset().adata.obs[dataset().spec.celltype_col].unique().tolist()
        )
        component_choices = [str(i) for i in range(1, cp_rank() + 1)]
        eigenstate_choices = [str(i) for i in range(1, rise_rank() + 1)]
        with reactive.isolate():
            cmp = str(input.component())
        return ui.TagList(
            ui.h4("PaCMAP Embeddings"),
            ui.markdown(
                "Three PaCMAP embeddings of individual cells for the "
                "selected component: weighted by sender or receiver score, "
                "colored by cell type, and colored by group/condition. Cell "
                "weight source switches the first panel between sender and "
                "receiver weight."
            ),
            ui.input_radio_buttons(
                "weight_source",
                "Cell weight source",
                choices={"sender": "Sender (B)", "receiver": "Receiver (C)"},
                selected="sender",
                inline=True,
            ),
            ui.output_plot("projections_plot", height="500px"),
            ui.hr(),
            ui.h4("Cell-Type Distributions"),
            ui.markdown(
                "Sender and receiver weight distributions for the selected "
                "component, by cell type. View Type switches between a "
                "histogram grid and boxplots; Log-scale Y axis applies to "
                "the histograms; Top Cell % Threshold sets the percentile "
                "marked on the histograms."
            ),
            ui.row(
                ui.column(
                    4,
                    ui.input_radio_buttons(
                        "dist_view_mode",
                        "View Type",
                        choices={
                            "hist": "Histogram Grid (Sender & Receiver Overlaid)",
                            "box": "Boxplots (Selected Component)",
                        },
                        selected="hist",
                    ),
                ),
                ui.column(
                    4,
                    ui.input_checkbox(
                        "dist_log_y", "Log-scale Y axis (Histograms)", value=True
                    ),
                ),
                ui.column(4, top_perc_slider("dist_top_perc")),
            ),
            scrollable_plot("distributions_plot", 1300, 650),
            ui.hr(),
            ui.h4("Cell-Type–Eigenstate Associations"),
            ui.markdown(
                "PaCMAP colored by each cell's projection onto the RISE "
                "eigenstate chosen with Eigenstate; opposite signs describe "
                "different cells. Color Scale Cap sets the percentile of "
                "|projection| the color scale spans; the few cells beyond it "
                "saturate. The cell type PaCMAP is repeated alongside for "
                "comparison."
            ),
            ui.row(
                ui.column(
                    6,
                    ui.input_slider(
                        "eigenstate",
                        "Eigenstate",
                        min=1,
                        max=rise_rank(),
                        value=1,
                        step=1,
                    ),
                ),
                ui.column(
                    6,
                    ui.input_slider(
                        "eig_span_percentile",
                        "Color Scale Cap (|projection| percentile)",
                        min=90,
                        max=100,
                        value=99,
                        step=0.5,
                    ),
                ),
            ),
            scrollable_plot("eigenstate_pacmap_plot", 1300, 550),
            ui.markdown(
                "Each cell type's association with every eigenstate, from "
                "the raw per-cell projections and independent of the "
                "selected component. Association Heatmap Stat selects the "
                "statistic; Top Cell % Threshold, shown only for the "
                "enrichment and % Top statistics, sets the percentile they "
                "use (both count only the positive tail)."
            ),
            ui.row(
                ui.column(
                    4,
                    ui.input_select(
                        "eig_assoc_stat",
                        "Association Heatmap Stat",
                        choices=ASSOC_STAT_CHOICES,
                        selected="mean",
                    ),
                ),
                ui.column(
                    4,
                    ui.panel_conditional(
                        f"{list(TOP_PERC_STATS)}.includes(input.eig_assoc_stat)",
                        top_perc_slider("eig_top_perc"),
                    ),
                ),
            ),
            scrollable_plot("eigenstate_assoc_plot", 1300, 500),
            ui.markdown(
                "Each cell's projection onto Eigenstate X plotted against "
                "Eigenstate Y, colored by cell type, with Highlighted Cell "
                "Type's points enlarged (by Selected Cell Type Point Size, "
                "below). Cells of one type loading on one eigenstate but not "
                "the other separate along the two axes, marking "
                "subpopulations within that cell type."
            ),
            ui.row(
                *(
                    ui.column(
                        4,
                        ui.input_select(
                            input_id, label, choices=eigenstate_choices, selected=value
                        ),
                    )
                    for input_id, label, value in (
                        ("eig_scatter_x", "Eigenstate X (x axis)", "1"),
                        ("eig_scatter_y", "Eigenstate Y (y axis)", "2"),
                    )
                ),
                ui.column(
                    4,
                    ui.input_select(
                        "eig_scatter_celltype",
                        "Highlighted Cell Type",
                        choices=celltypes,
                    ),
                ),
            ),
            scrollable_plot("eigenstate_scatter_plot", *SCATTER_SINGLE_PX),
            ui.hr(),
            ui.h4("Sender/Receiver Classification"),
            ui.markdown(
                "A cell counts as a strong sender (or receiver) if its "
                "weight for the selected component is at or above the "
                "chosen percentile across all cells. Cell Type selects the "
                "cell type counted in the truth table and enlarged in the "
                "PaCMAP and scatter below; Strength Threshold (percentile) "
                "sets that percentile for all three panels below; Selected "
                "Cell Type Point Size sets how much larger (in diameter) "
                "that cell type's points are drawn."
            ),
            ui.row(
                ui.column(
                    4,
                    ui.input_select("tt_celltype", "Cell Type", choices=celltypes),
                ),
                ui.column(
                    4,
                    ui.input_slider(
                        "tt_percentile",
                        "Strength Threshold (percentile)",
                        min=50,
                        max=99,
                        value=90,
                        step=1,
                    ),
                ),
                ui.column(
                    4,
                    ui.input_slider(
                        "tt_point_size",
                        "Selected Cell Type Point Size (% increase)",
                        min=100,
                        max=300,
                        value=250,
                        step=10,
                    ),
                ),
            ),
            ui.markdown(
                "###### Truth Table (Selected Cell Type)\n"
                "Counts of the selected cell type's cells that are a "
                "strong sender, strong receiver, both, or neither, with "
                "row/column totals."
            ),
            ui.output_data_frame("truth_table"),
            ui.markdown(
                "###### Strength Category PaCMAP (All Cell Types)\n"
                "PaCMAP embedding colored by strength category (Both, "
                "Sender Only, Receiver Only, Neither), for every cell in "
                "the dataset, with the selected cell type's points "
                "enlarged."
            ),
            scrollable_plot("strength_category_pacmap_plot", 650, 550),
            ui.markdown(
                "###### Sender vs. Receiver Weight (All Cell Types)\n"
                "Per-cell weights for Component X (x axis) against "
                "Component Y (y axis), colored by cell type, with the "
                "selected cell type's points enlarged. Both default to (and "
                "follow) the sidebar CP Component. When they match, a single "
                "panel plots that component's sender against receiver "
                "weight; when they differ, a 2x2 grid compares every role "
                "pairing (rows: X sender/receiver; columns: Y "
                "sender/receiver). Dashed lines mark the current Strength "
                "Threshold percentile."
            ),
            ui.row(
                *(
                    ui.column(
                        4,
                        ui.input_select(
                            input_id, label, choices=component_choices, selected=cmp
                        ),
                    )
                    for input_id, label in SCATTER_COMPONENT_INPUTS.items()
                )
            ),
            ui.output_ui("sender_receiver_scatter_ui"),
        )

    def scatter_components() -> tuple[int, int]:
        return int(input.scatter_x_cmp()), int(input.scatter_y_cmp())

    @render.ui
    def sender_receiver_scatter_ui():
        x_cmp, y_cmp = scatter_components()
        size = SCATTER_SINGLE_PX if x_cmp == y_cmp else SCATTER_GRID_PX
        return scrollable_plot("sender_receiver_scatter_plot", *size)

    @reactive.effect
    @reactive.event(input.component)
    def _sync_scatter_components():
        for input_id in SCATTER_COMPONENT_INPUTS:
            ui.update_select(input_id, selected=str(input.component()))

    def weights_and_threshold(obsm_key: str, cmp: int) -> tuple[np.ndarray, float]:
        weights = np.asarray(dataset().adata.obsm[obsm_key])[:, cmp - 1]
        return weights, float(np.percentile(weights, input.tt_percentile()))

    @reactive.calc
    def strength_weights():
        """(sender weights, receiver weights, sender threshold, receiver
        threshold) of the selected component."""
        cmp = input.component()
        sender, sender_threshold = weights_and_threshold("sc_B", cmp)
        receiver, receiver_threshold = weights_and_threshold("rc_C", cmp)
        return sender, receiver, sender_threshold, receiver_threshold

    @reactive.calc
    def selected_celltype_mask():
        ds = dataset()
        return ds.adata.obs[ds.spec.celltype_col].to_numpy() == input.tt_celltype()

    @render.data_frame
    def truth_table():
        sender, receiver, sender_threshold, receiver_threshold = strength_weights()

        in_celltype = selected_celltype_mask()
        is_sender = (sender >= sender_threshold) & in_celltype
        is_receiver = (receiver >= receiver_threshold) & in_celltype

        both = int(np.sum(is_sender & is_receiver))
        sender_only = int(np.sum(is_sender & ~is_receiver))
        receiver_only = int(np.sum(~is_sender & is_receiver))
        neither = int(np.sum(~is_sender & ~is_receiver & in_celltype))

        table = pd.DataFrame(
            {
                "": ["Strong Receiver", "Not Strong Receiver", "Total"],
                "Strong Sender": [both, sender_only, both + sender_only],
                "Not Strong Sender": [
                    receiver_only,
                    neither,
                    receiver_only + neither,
                ],
                "Total": [
                    both + receiver_only,
                    sender_only + neither,
                    int(np.sum(in_celltype)),
                ],
            }
        )
        # Style the row labels like the header row
        return render.DataGrid(
            table,
            styles=[
                {
                    "cols": [0],
                    "style": {
                        "background-color": "var(--shiny-datagrid-grid-header-bgcolor)",
                        "font-weight": "bold",
                    },
                }
            ],
        )

    @render.plot
    @styled_figure()
    def strength_category_pacmap_plot():
        ds = dataset()
        sender, receiver, sender_threshold, receiver_threshold = strength_weights()
        is_sender = sender >= sender_threshold
        is_receiver = receiver >= receiver_threshold
        categories = np.where(
            is_sender & is_receiver,
            "Both",
            np.where(
                is_sender & ~is_receiver,
                "Sender Only",
                np.where(~is_sender & is_receiver, "Receiver Only", "Neither"),
            ),
        )

        points = np.asarray(ds.adata.obsm["PaCMAP"])
        fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="tight")
        plot_categorical_scatter(ax, points, categories, STRENGTH_CATEGORY_COLORS)
        selected = np.flatnonzero(selected_celltype_mask())
        # Strong senders/receivers drawn last, over the "Neither" majority
        selected = selected[
            np.argsort(categories[selected] != "Neither", kind="stable")
        ]
        enlarge_points(
            ax,
            points[selected],
            [STRENGTH_CATEGORY_COLORS[c] for c in categories[selected]],
            scale=1 + input.tt_point_size() / 100,
        )
        pacmap_axes(ax)
        return fig

    @render.plot
    @styled_figure()
    def sender_receiver_scatter_plot():
        x_cmp, y_cmp = scatter_components()
        if x_cmp == y_cmp:
            role_grid = [[("Sender", "Receiver")]]
        else:
            role_grid = [
                [(xr, yr) for yr in ROLE_WEIGHT_KEYS] for xr in ROLE_WEIGHT_KEYS
            ]
        n = len(role_grid)
        width_px, height_px = SCATTER_SINGLE_PX if n == 1 else SCATTER_GRID_PX
        fig, axes = plt.subplots(
            n,
            n,
            figsize=(width_px / 100, height_px / 100),
            squeeze=False,
            layout="tight",
        )
        for ax_row, role_row in zip(axes, role_grid, strict=True):
            for ax, (x_role, y_role) in zip(ax_row, role_row, strict=True):
                plot_weight_scatter(ax, (x_cmp, x_role), (y_cmp, y_role))
        # One shared cell type legend, on the top-right panel
        for i, ax in enumerate(axes.flat):
            if i != n - 1:
                ax.get_legend().remove()
        return fig

    def plot_weight_scatter(
        ax: plt.Axes, x: tuple[int, str], y: tuple[int, str]
    ) -> None:
        """Cells' (component, role) weights against each other, with the
        selected cell type enlarged and Strength Threshold lines."""
        (x_weights, x_threshold), (y_weights, y_threshold) = (
            weights_and_threshold(ROLE_WEIGHT_KEYS[role], cmp) for cmp, role in (x, y)
        )
        plot_celltype_scatter(
            ax,
            np.column_stack([x_weights, y_weights]),
            input.tt_celltype(),
            thresholds=(x_threshold, y_threshold),
        )
        ax.set(xlabel=f"Cmp. {x[0]} {x[1]} Weight", ylabel=f"Cmp. {y[0]} {y[1]} Weight")

    def plot_celltype_scatter(
        ax: plt.Axes,
        points: np.ndarray,
        celltype: str,
        thresholds: tuple[float, float] | None = None,
    ) -> None:
        """Per-cell 2D `points` colored by cell type, `celltype`'s points
        enlarged on top, and optional (x, y) dashed threshold lines."""
        ds = dataset()
        plot_celltype_points(
            ax,
            ds.adata,
            ds.spec.key,
            points,
            selected=celltype,
            selected_scale=1 + input.tt_point_size() / 100,
        )
        if thresholds is not None:
            ax.axvline(thresholds[0], color="0.3", ls="--", lw=1.0)
            ax.axhline(thresholds[1], color="0.3", ls="--", lw=1.0)

    @render.plot
    @styled_figure()
    def eigenstate_scatter_plot():
        projections = np.asarray(dataset().adata.obsm["projections"])
        x_eig, y_eig = int(input.eig_scatter_x()), int(input.eig_scatter_y())
        width_px, height_px = SCATTER_SINGLE_PX
        fig, ax = plt.subplots(
            figsize=(width_px / 100, height_px / 100), layout="tight"
        )
        plot_celltype_scatter(
            ax, projections[:, [x_eig - 1, y_eig - 1]], input.eig_scatter_celltype()
        )
        ax.set(
            xlabel=f"Eigenstate {x_eig} Projection",
            ylabel=f"Eigenstate {y_eig} Projection",
        )
        return fig

    @render.plot
    @styled_figure()
    def projections_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        weights = X.obsm[ROLE_WEIGHT_KEYS[input.weight_source().capitalize()]]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), layout="tight")

        plot_value_pacmap(
            X,
            weights[:, cmp - 1],
            axes[0],
            f"Cmp. {cmp} {input.weight_source().capitalize()} Weight",
            "Weight",
            SATURATION_PERCENTILE,
        )

        plot_celltype_pacmap(ds, axes[1])

        if ds.spec.group_col is not None and ds.spec.group_col in X.obs:
            groups = X.obs[ds.spec.group_col].astype(str).to_numpy()
            levels = np.unique(groups)
            plot_categorical_scatter(
                axes[2],
                np.asarray(X.obsm["PaCMAP"]),
                groups,
                dict(zip(levels, colors.categorical_colors(len(levels)), strict=True)),
            )
            pacmap_axes(axes[2])
            axes[2].set_title(ds.spec.group_col)
        else:
            axes[2].axis("off")

        return fig

    @render.plot
    @styled_figure()
    def eigenstate_pacmap_plot():
        ds = dataset()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), layout="tight")
        eigenstate = input.eigenstate()
        plot_value_pacmap(
            ds.adata,
            np.asarray(ds.adata.obsm["projections"])[:, eigenstate - 1],
            axes[0],
            f"Eigenstate {eigenstate} Weight",
            "Projection",
            input.eig_span_percentile(),
        )
        plot_celltype_pacmap(ds, axes[1])
        return fig

    @render.plot
    @styled_figure()
    def eigenstate_assoc_plot():
        ds = dataset()
        fig, ax = plt.subplots(figsize=(16, 6), layout="tight")
        plot_celltype_association(
            ds.adata,
            ax,
            factor_type="eigenstate",
            celltype_col=ds.spec.celltype_col,
            stat=input.eig_assoc_stat(),
            top_perc=float(input.eig_top_perc()),
        )
        ax.set_title("Cell Type Association with Eigenstates")

        return fig

    # ------------------------------------------------------------------
    # Summary Tables
    # ------------------------------------------------------------------
    @render.ui
    def summary_tables_body():
        if not dataset().has_projections:
            return ui.markdown(
                "Summary tables are unavailable for this dataset "
                "(raw single-cell data or projections not found)."
            )
        return ui.TagList(
            ui.markdown("#### Cell Type Loadings Summary"),
            ui.markdown(
                "Mean sender/receiver loading and % of cells in the top "
                "percentile, by cell type, for the selected component. Top "
                "Cell % Threshold sets the percentile used by the '% of "
                "Cells in Top' columns; mean loading is unaffected by it."
            ),
            top_perc_slider("summary_top_perc"),
            ui.output_data_frame("celltype_summary_table"),
            ui.hr(),
            ui.markdown("#### Ranked Ligand-Receptor Pairs"),
            ui.markdown(
                "Every LR pair's weight for the selected component, ranked "
                "from highest to lowest."
            ),
            ui.output_data_frame("lr_ranked_table"),
        )

    @render.data_frame
    def celltype_summary_table():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        top_perc = float(input.summary_top_perc())

        stat_labels = {
            "mean": "Mean Loading",
            "pct_top": f"% of Cells in Top {top_perc:g}%",
        }
        summary = pd.concat(
            {
                f"{factor_type.capitalize()} {stat_labels[stat]}": celltype_component_association(
                    X,
                    factor_type=factor_type,
                    celltype_col=celltype_col,
                    stat=stat,
                    top_perc=top_perc,
                )[cmp]
                for factor_type in ("sender", "receiver")
                for stat in ("mean", "pct_top")
            },
            axis=1,
        )
        summary = summary.reset_index().rename(columns={"index": "Cell Type"})
        summary = summary.round(4)
        return render.DataGrid(summary)

    @render.data_frame
    def lr_ranked_table():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        weights = np.asarray(X.uns["D"], dtype=float)[:, cmp - 1]
        names = np.asarray(X.uns["lr_pairs"]).astype(str)
        order = np.argsort(-weights)
        ranked = pd.DataFrame(
            {
                "Rank": np.arange(1, weights.size + 1),
                "LR Pair": names[order],
                "Weight": weights[order].round(4),
            }
        )
        return render.DataGrid(ranked)


app = App(app_ui, server)
