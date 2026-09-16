"""Shiny app for interactively exploring RISE / CCC-RISE decomposition output.

Run with:
    shiny run --reload cccrise/app/app.py
    shiny run --host 0.0.0.0 --port 8000 --reload cccrise/app/app.py
"""

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
    plot_logreg_weights,
    plot_lr_factor_rank,
    plot_lr_factors,
)
from cccrise.figures.commonFuncs.plotGeneral import (
    binarize_group,
    celltype_grid_shape,
    get_sample_to_group,
    plot_wc_histogram_grid,
)
from cccrise.figures.commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
    plot_wc_pacmap,
    plot_wc_per_celltype,
)
from cccrise.logreg import fit_logreg_weights
from cccrise.utils import celltype_component_association

DATASET_CHOICES = {key: spec.label for key, spec in DATASETS.items()}

LR_HIGHLIGHT_CHOICES = {
    "5": "Top 5 Pairs",
    "10": "Top 10 Pairs",
    "15": "Top 15 Pairs",
    "tukey": "Tukey Outliers (Q3 + 1.5*IQR)",
    "q3": "Q3 (75th Percentile)",
}


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
        ui.input_slider(
            "top_perc",
            "Top Cell % Threshold",
            min=1,
            max=50,
            value=10,
            step=1,
        ),
        ui.output_ui("projections_notice"),
        width=300,
    ),
    ui.navset_tab(
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
                "⚠️ This dataset's factors file has no stored per-cell "
                "projections; only factor matrices are available. Regenerate "
                "it with `cccrise.regenerate_factors`."
            )
        return None

    # ------------------------------------------------------------------
    # Factor Matrices
    # ------------------------------------------------------------------
    @render.plot
    def factor_heatmaps():
        ds = dataset()
        X = ds.adata
        fig, axes = plt.subplots(1, 4, figsize=(20, 8), layout="constrained")

        plot_condition_factors(X, axes[0], cond=ds.spec.condition_col)
        axes[0].set_title("Condition Factor (A)")

        plot_eigenstate_factors(X, axes[1], factor_type="B")
        axes[1].set_title("Sender Eigenstate Factor (B)")

        plot_eigenstate_factors(X, axes[2], factor_type="C")
        axes[2].set_title("Receiver Eigenstate Factor (C)")

        plot_lr_factors(X, axes[3], trim=True, weight=input.lr_weight())
        axes[3].set_title("Ligand-Receptor Factor (D)")

        return fig

    # ------------------------------------------------------------------
    # Component Explorer (top LR pairs, per-cell-type distributions, and
    # the single-pair interaction heatmap, all keyed on the sidebar's
    # dataset/component/top-cell-% selections)
    # ------------------------------------------------------------------
    @reactive.calc
    def simple_lr_pairs():
        ds = dataset()
        pairs = parse_lr_pairs(np.asarray(ds.adata.uns["lr_pairs"]))
        return [p for p in pairs if p.is_simple_receptor]

    @reactive.calc
    def logreg_result() -> tuple[np.ndarray | None, str | None]:
        ds = dataset()
        spec = ds.spec
        if not ds.has_projections:
            return None, "per-cell metadata needed to derive the group label is unavailable."
        if spec.control_label is None or spec.group_col is None:
            return None, "no case/control grouping is configured for this dataset."
        try:
            sample_to_group = get_sample_to_group(
                ds.adata, spec.condition_col, spec.group_col
            )
            binary = binarize_group(
                sample_to_group, spec.control_label, spec.control_label, spec.other_label
            )
            weights = fit_logreg_weights(ds.adata, binary, scoring="roc_auc")
        except ValueError as exc:
            return None, str(exc)
        return weights, None

    @render.ui
    def logreg_weights_ui():
        _, error = logreg_result()
        if error is not None:
            return ui.markdown(f"_Logistic regression weights unavailable: {error}_")
        return ui.TagList(
            ui.output_plot("logreg_weights_plot", height="220px"),
            ui.markdown(
                "_Fit live on this dataset's stored condition factors "
                f"({dataset().spec.control_label!r} vs. "
                f"{dataset().spec.other_label!r}); unlike the manuscript "
                "figures, this skips read-depth correction since the app "
                "doesn't keep raw counts on hand._"
            ),
        )

    @render.plot
    def logreg_weights_plot():
        weights, _ = logreg_result()
        if weights is None:
            return None
        fig, ax = plt.subplots(figsize=(10, 2.4), layout="constrained")
        plot_logreg_weights(weights, ax, highlight_cmp=input.component())
        return fig

    @render.ui
    def explorer_body():
        ds = dataset()
        if not ds.has_projections:
            return ui.markdown(
                "Per-cell weights are unavailable for this dataset "
                "(raw single-cell data or projections not found)."
            )

        sections = [
            ui.h4("Top Ligand-Receptor Pairs"),
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
            ui.output_ui("logreg_weights_ui"),
            scrollable_plot("lr_rank_curve_plot", 1100, 450),
            ui.hr(),
            ui.h4("Per-Cell-Type Distributions"),
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
                    ui.input_checkbox(
                        "dist_log_y", "Log-scale Y axis (Histograms)", value=True
                    ),
                ),
                ui.column(
                    4,
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
            scrollable_plot("distributions_plot", 1300, 650),
            ui.hr(),
            ui.h4("LR Pair Interaction Heatmap"),
        ]

        raw = load_raw_dataset(input.dataset())
        if raw.error is not None or raw.adata is None:
            sections.append(ui.markdown(f"⚠️ {raw.error or 'Raw dataset unavailable.'}"))
            return ui.TagList(*sections)

        celltypes = sorted(raw.adata.obs[ds.spec.celltype_col].unique().tolist())
        pair_labels = [p.label for p in simple_lr_pairs()]

        sections.extend(
            [
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
    def distributions_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        view_mode = input.dist_view_mode()
        top_perc = float(input.top_perc())

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
                top_perc=top_perc,
                log_y=input.dist_log_y(),
            )
            fig.suptitle(f"Cmp. {cmp}: Cell Weight Distributions by Cell Type")
            return fig

        elif view_mode == "assoc":
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
            stat = input.dist_assoc_stat()

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
            ui.markdown("#### Cell Type Loadings Summary for Selected Component"),
            ui.output_data_frame("celltype_summary_table"),
            ui.hr(),
            ui.markdown("#### Ranked Ligand-Receptor Pairs for Selected Component"),
            ui.output_data_frame("lr_ranked_table"),
        )

    @render.data_frame
    def celltype_summary_table():
        ds = dataset()
        X = ds.adata
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        top_perc = float(input.top_perc())

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
