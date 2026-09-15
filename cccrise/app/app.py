"""Shiny app for interactively exploring RISE / CCC-RISE decomposition output.

Run with:
    shiny run --reload cccrise/app/app.py
    shiny run --host 0.0.0.0 --port 8000 --reload cccrise/app/app.py
"""

import math
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

from cccrise.app.data import DATASETS, load_dataset, load_raw_dataset
from cccrise.app.lr_pairs import parse_lr_pairs
from cccrise.figures.commonFuncs.plotFactors import (
    plot_celltype_association,
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_interaction_heatmap,
    plot_lr_factor_rank,
    plot_lr_factors,
    plot_lr_factors_partial,
)
from cccrise.figures.commonFuncs.plotGeneral import (
    celltype_grid_shape,
    plot_wc_histogram_grid,
)
from cccrise.figures.commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
    plot_wc_pacmap,
    plot_wc_per_celltype,
)
from cccrise.utils import celltype_component_association

DATASET_CHOICES = {key: spec.label for key, spec in DATASETS.items()}

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("dataset", "Dataset", choices=DATASET_CHOICES),
        ui.input_slider("component", "CP Component", min=1, max=1, value=1, step=1),
        ui.input_radio_buttons(
            "weight_source",
            "Cell weight source (PaCMAP)",
            choices={"sender": "Sender (B)", "receiver": "Receiver (C)"},
            selected="sender",
        ),
        ui.output_ui("projections_notice"),
        width=300,
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Component Interpretation",
            ui.output_ui("interpretation_body"),
        ),
        ui.nav_panel(
            "LR Explorer",
            ui.output_ui("lr_explorer_body"),
        ),
        ui.nav_panel(
            "Per-Cell-Type Distributions",
            ui.output_ui("distributions_body"),
        ),
        ui.nav_panel(
            "Cell Projections",
            ui.output_ui("projections_body"),
        ),
        ui.nav_panel(
            "Factor Matrices",
            ui.input_slider(
                "lr_weight",
                "LR factor trim threshold (min |weight|)",
                min=0,
                max=0.3,
                value=0.06,
                step=0.01,
            ),
            ui.output_plot("factor_heatmaps", height="650px"),
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
    def _sync_component_slider():
        rank = cp_rank()
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
                "⚠️ This dataset's factors file has no stored per-cell "
                "projections; only factor matrices are available. Regenerate "
                "it with `cccrise.regenerate_factors`."
            )
        return None

    # ------------------------------------------------------------------
    # Component Interpretation (Fig A6i unified panel)
    # ------------------------------------------------------------------
    @render.ui
    def interpretation_body():
        if not dataset().has_projections:
            return ui.markdown(
                "Per-cell weights are unavailable for this dataset "
                "(raw single-cell data or projections not found)."
            )
        return ui.TagList(
            ui.row(
                ui.column(
                    4,
                    ui.input_select(
                        "interp_highlight",
                        "LR Highlight Rule",
                        choices={
                            "5": "Top 5 Pairs",
                            "10": "Top 10 Pairs",
                            "15": "Top 15 Pairs",
                            "tukey": "Tukey Outliers (Q3 + 1.5*IQR)",
                            "q3": "Q3 (75th Percentile)",
                        },
                        selected="5",
                    ),
                ),
                ui.column(
                    4,
                    ui.input_slider(
                        "interp_top_perc",
                        "Top Cell % Threshold",
                        min=1,
                        max=50,
                        value=10,
                        step=1,
                    ),
                ),
                ui.column(
                    4,
                    ui.input_checkbox(
                        "interp_include_negative",
                        "Include Negative LR Tail",
                        value=True,
                    ),
                ),
            ),
            ui.output_plot("interpretation_plot", height="900px"),
            ui.hr(),
            ui.markdown("#### Cell Type Loadings Summary for Selected Component"),
            ui.output_data_frame("interp_summary_table"),
        )

    @render.plot
    def interpretation_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col

        highlight_raw = input.interp_highlight()
        highlight: int | str = (
            int(highlight_raw) if highlight_raw.isdigit() else highlight_raw
        )
        include_neg = input.interp_include_negative()
        top_perc = float(input.interp_top_perc())

        nrows, ncols = celltype_grid_shape(X, celltype_col)
        grid_width = 4.5 * ncols
        grid_height = 3.2 * nrows
        lr_height = 3.5

        fig = plt.figure(
            figsize=(grid_width, grid_height + lr_height),
            layout="constrained",
        )
        top, bottom = fig.subfigures(2, 1, height_ratios=[lr_height, grid_height])

        lr_width = min(6.0, grid_width)
        if lr_width < grid_width:
            ax_lr = top.add_subplot(
                top.add_gridspec(1, 2, width_ratios=[lr_width, grid_width - lr_width])[
                    0, 0
                ]
            )
        else:
            ax_lr = top.subplots()

        plot_lr_factor_rank(
            X,
            cmp,
            ax_lr,
            highlight=highlight,
            include_negative=include_neg,
        )

        plot_wc_histogram_grid(
            X,
            cmp,
            bottom.subplots(nrows, ncols),
            celltype_col=celltype_col,
            top_perc=top_perc,
        )
        bottom.suptitle("Cell Weight Distributions by Cell Type")
        fig.suptitle(f"Component {cmp} Interpretation", fontweight="bold")

        return fig

    @render.data_frame
    def interp_summary_table():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        top_perc = float(input.interp_top_perc())

        summary = pd.concat(
            {
                f"{factor_type}_{stat}": celltype_component_association(
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

    # ------------------------------------------------------------------
    # Per-Cell-Type Distributions
    # ------------------------------------------------------------------
    @render.ui
    def distributions_body():
        if not dataset().has_projections:
            return ui.markdown(
                "Per-cell weights are unavailable for this dataset "
                "(raw single-cell data not found)."
            )
        return ui.TagList(
            ui.row(
                ui.column(
                    4,
                    ui.input_radio_buttons(
                        "dist_view_mode",
                        "View Type",
                        choices={
                            "hist": "Histogram Grid (Sender & Receiver Overlaid)",
                            "assoc": "Cell Type Association Heatmap (All Components)",
                            "box": "Boxplots (Selected Component)",
                        },
                        selected="hist",
                    ),
                ),
                ui.column(
                    4,
                    ui.input_slider(
                        "dist_top_perc",
                        "Top Cell % Threshold",
                        min=1,
                        max=50,
                        value=10,
                        step=1,
                    ),
                ),
                ui.column(
                    4,
                    ui.input_checkbox(
                        "dist_log_y", "Log-scale Y axis (Histograms)", value=True
                    ),
                    ui.input_select(
                        "dist_assoc_stat",
                        "Association Heatmap Stat",
                        choices={
                            "mean": "Mean Cell Loading",
                            "effect_size": "Cohen's d vs Other Cells",
                            "enrichment": "log2 Enrichment",
                            "pct_top": "% in Top Percentile",
                        },
                        selected="mean",
                    ),
                ),
            ),
            ui.output_plot("distributions_plot", height="650px"),
            ui.hr(),
            ui.markdown("#### Cell Type Summary Table"),
            ui.output_data_frame("distributions_summary_table"),
        )

    @render.plot
    def distributions_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        view_mode = input.dist_view_mode()

        if view_mode == "hist":
            nrows, ncols = celltype_grid_shape(X, celltype_col)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), layout="constrained"
            )
            plot_wc_histogram_grid(
                X,
                cmp,
                axes,
                celltype_col=celltype_col,
                top_perc=float(input.dist_top_perc()),
                log_y=input.dist_log_y(),
            )
            fig.suptitle(f"Cmp. {cmp}: Cell Weight Distributions by Cell Type")
            return fig

        elif view_mode == "assoc":
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
            stat = input.dist_assoc_stat()
            top_perc = float(input.dist_top_perc())

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
                    normalize="global",
                    scale=scale,
                    show_counts=True,
                )
                ax.set_title(f"{label} Cell Type Association")
            fig.suptitle(f"Cell Type Associations ({stat})", fontweight="bold")
            return fig

        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
            plot_wc_per_celltype(
                X, cmp, axes[0], cellType=celltype_col, factor_matrix="B"
            )
            axes[0].set_title(f"Cmp. {cmp} Sender Weight by Cell Type")

            plot_wc_per_celltype(
                X, cmp, axes[1], cellType=celltype_col, factor_matrix="C"
            )
            axes[1].set_title(f"Cmp. {cmp} Receiver Weight by Cell Type")
            return fig

    @render.data_frame
    def distributions_summary_table():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        top_perc = float(input.dist_top_perc())

        summary = pd.concat(
            {
                f"{factor_type}_{stat}": celltype_component_association(
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

    # ------------------------------------------------------------------
    # LR Explorer
    # ------------------------------------------------------------------
    @reactive.calc
    def simple_lr_pairs():
        ds = dataset()
        pairs = parse_lr_pairs(np.asarray(ds.adata.uns["lr_pairs"]))
        return [p for p in pairs if p.is_simple_receptor]

    @render.ui
    def lr_explorer_body():
        ds = dataset()
        rank_panel = ui.TagList(
            ui.row(
                ui.column(
                    4,
                    ui.input_select(
                        "lr_highlight_mode",
                        "Highlight Rule",
                        choices={
                            "5": "Top 5 Pairs",
                            "10": "Top 10 Pairs",
                            "15": "Top 15 Pairs",
                            "tukey": "Tukey Outliers (Q3 + 1.5*IQR)",
                            "q3": "Q3 (75th Percentile)",
                        },
                        selected="5",
                    ),
                ),
                ui.column(
                    4,
                    ui.input_checkbox(
                        "lr_include_negative",
                        "Include Negative LR Tail",
                        value=True,
                    ),
                ),
                ui.column(
                    4,
                    ui.input_numeric(
                        "lr_max_labels", "Max Callout Labels", value=10, min=1, max=30
                    ),
                ),
            ),
            ui.output_plot("lr_rank_curve_plot", height="400px"),
            ui.output_plot("lr_top_bottom_plot", height="300px"),
            ui.hr(),
            ui.markdown("#### Ranked Ligand-Receptor Pairs for Selected Component"),
            ui.output_data_frame("lr_ranked_table"),
        )

        raw = load_raw_dataset(input.dataset())
        if raw.error is not None:
            return ui.TagList(rank_panel, ui.markdown(f"⚠️ {raw.error}"))

        celltypes = sorted(raw.adata.obs[ds.spec.celltype_col].unique().tolist())
        pair_labels = [p.label for p in simple_lr_pairs()]

        return ui.TagList(
            rank_panel,
            ui.hr(),
            ui.markdown("### Single-Pair Interaction Heatmap (Expression Product)"),
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
            ui.output_plot("lr_interaction_plot", height="450px"),
        )

    @render.plot
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

        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        plot_lr_factor_rank(
            X,
            cmp,
            ax,
            highlight=highlight,
            include_negative=include_neg,
            max_labels=max_labels,
        )
        return fig

    @render.plot
    def lr_top_bottom_plot():
        ds = dataset()
        cmp = input.component()
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), layout="constrained")
        plot_lr_factors_partial(ds.adata, cmp, axes[0], geneAmount=10, top=True)
        axes[0].set_title(f"Cmp. {cmp}: Top LR Pairs")
        plot_lr_factors_partial(ds.adata, cmp, axes[1], geneAmount=10, top=False)
        axes[1].set_title(f"Cmp. {cmp}: Bottom LR Pairs")
        return fig

    @render.data_frame
    def lr_ranked_table():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        highlight_raw = input.lr_highlight_mode()
        highlight: int | str = (
            int(highlight_raw) if highlight_raw.isdigit() else highlight_raw
        )
        include_neg = input.lr_include_negative()
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

    @render.plot
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
        if not dataset().has_projections:
            return ui.markdown(
                "Per-cell projections are unavailable for this dataset "
                "(raw single-cell data not found)."
            )
        return ui.output_plot("projections_plot", height="500px")

    @render.plot
    def projections_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        factor_matrix = "B" if input.weight_source() == "sender" else "C"

        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

        plot_wc_pacmap(X, cmp, axes[0], cbarMax=0.3, factor_matrix=factor_matrix)
        axes[0].set_title(f"Cmp. {cmp} {input.weight_source().capitalize()} Weight")

        plot_labels_pacmap(X, labelType=ds.spec.celltype_col, ax=axes[1])
        axes[1].set_title("Cell Type")

        if ds.spec.group_col is not None and ds.spec.group_col in X.obs:
            plot_labels_pacmap(X, labelType=ds.spec.group_col, ax=axes[2])
            axes[2].set_title(ds.spec.group_col)
        else:
            axes[2].axis("off")

        return fig

    # ------------------------------------------------------------------
    # Factor Matrices
    # ------------------------------------------------------------------
    @render.plot
    def factor_heatmaps():
        ds = dataset()
        X = ds.adata
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))

        plot_condition_factors(X, axes[0], cond=ds.spec.condition_col)
        axes[0].set_title("Condition Factor (A)")

        plot_eigenstate_factors(X, axes[1], factor_type="B")
        axes[1].set_title("Sender Eigenstate Factor (B)")

        plot_eigenstate_factors(X, axes[2], factor_type="C")
        axes[2].set_title("Receiver Eigenstate Factor (C)")

        plot_lr_factors(X, axes[3], trim=True, weight=input.lr_weight())
        axes[3].set_title("Ligand-Receptor Factor (D)")

        return fig


app = App(app_ui, server)
