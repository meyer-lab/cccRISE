"""Shiny app for interactively exploring RISE / CCC-RISE decomposition output.

Run with:
    shiny run --reload cccrise/app/app.py
    shiny run --host 0.0.0.0 --port 8000 --reload cccrise/app/app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

from cccrise.anova import run_saved_factors_anova
from cccrise.app.data import DATASETS, load_dataset, load_raw_dataset
from cccrise.app.lr_pairs import parse_lr_pairs
from cccrise.figures.commonFuncs.plotFactors import (
    plot_celltype_association,
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_interaction_heatmap,
    plot_lr_factor_rank,
    plot_lr_factors,
)
from cccrise.figures.commonFuncs.plotGeneral import (
    celltype_grid_shape,
    plot_wc_histogram_grid,
)
from cccrise.figures.commonFuncs.plotPaCMAP import (
    assign_labels,
    plot_categorical_scatter,
    plot_labels_pacmap,
    plot_wc_pacmap,
    plot_wc_per_celltype,
)
from cccrise.figures.figureAnova import plot_association_effect_sizes
from cccrise.utils import celltype_component_association

DATASET_CHOICES = {key: spec.label for key, spec in DATASETS.items()}

LR_HIGHLIGHT_CHOICES = {
    "5": "Top 5 Pairs",
    "10": "Top 10 Pairs",
    "15": "Top 15 Pairs",
    "tukey": "Tukey Outliers (Q3 + 1.5*IQR)",
    "q3": "Q3 (75th Percentile)",
}

# `plot_categorical_scatter` sorts categories alphabetically for its color
# assignment, so this list has to be in that order: Both, Neither, Receiver
# Only, Sender Only. "Neither" is usually the majority category, so it's
# muted rather than drawn from the same accented palette as the other three.
STRENGTH_CATEGORY_COLORS = ["#7B3294", "#D9D9D9", "#E66101", "#2C7BB6"]


def _fit_anova_result(spec) -> tuple[pd.DataFrame | None, str | None]:
    """One dataset's per-component ANOVA of condition loading against its
    case/control grouping (or why it's unavailable).

    Uses `run_saved_factors_anova` exactly as `figureAnova.py` does: it reads
    condition/group metadata and factor matrices straight off the dataset's
    own per-cell obs (already reconstructed from the `.nc` file by
    `load_dataset`), so unlike the logistic regression it replaced, this
    needs no raw single-cell data and no read-depth correction.
    """
    if spec.control_label is None or spec.group_col is None:
        return None, "no case/control grouping is configured for this dataset."
    ds = load_dataset(spec.key)
    if not ds.has_projections:
        return (
            None,
            "per-cell metadata needed to derive the group label is unavailable.",
        )
    try:
        df = run_saved_factors_anova(
            factors=ds.adata,
            condition_column=spec.condition_col,
            group_column=spec.group_col,
            control_label=spec.control_label,
            other_label=spec.other_label,
            correction="fdr_bh",
        )
    except ValueError as exc:
        return None, str(exc)
    return df, None


# There's no saved ANOVA result to load -- like the logistic regression it
# replaced, every fit is ad hoc -- so it's run once per dataset here, when
# the app boots, rather than lazily per-session: unlike the factorization
# (loaded, never recomputed, from each dataset's `.nc` file) this needs to
# reflect whatever the current data says on every restart.
ANOVA_RESULTS: dict[str, tuple[pd.DataFrame | None, str | None]] = {
    key: _fit_anova_result(spec) for key, spec in DATASETS.items()
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
        fig, axes = plt.subplots(1, 4, figsize=(20, 8), layout="tight")

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

    def anova_result() -> tuple[pd.DataFrame | None, str | None]:
        # Computed once at boot for every dataset -- see ANOVA_RESULTS --
        # not re-fit per session or per input change.
        return ANOVA_RESULTS.get(
            input.dataset(), (None, "no ANOVA result for this dataset.")
        )

    @render.ui
    def anova_associations_ui():
        _, error = anova_result()
        if error is not None:
            return ui.markdown(f"_ANOVA component associations unavailable: {error}_")
        return ui.TagList(
            ui.output_plot("anova_associations_plot", height="260px"),
            ui.markdown(
                "_Fit once at app startup, as in the manuscript figures, on "
                f"{dataset().spec.control_label!r} vs. "
                f"{dataset().spec.other_label!r}, FDR-adjusted across "
                "components; not re-fit while the app is running._"
            ),
        )

    @render.plot
    def anova_associations_plot():
        df, _ = anova_result()
        if df is None:
            return None
        fig, ax = plt.subplots(figsize=(10, 2.8), layout="tight")
        plot_association_effect_sizes(df, ax)
        cmp = input.component()
        ax.axvline(cmp - 1, color="0.15", ls="--", lw=1.2, zorder=0)
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
            ui.h4("Component Association (ANOVA)"),
            ui.output_ui("anova_associations_ui"),
            ui.hr(),
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
            ),
            scrollable_plot("distributions_plot", 1300, 650),
            ui.hr(),
            ui.h4("Cell Type Association Heatmap"),
            ui.markdown(
                "How each cell type associates with *every* component at "
                "once -- a bird's-eye complement to the single-component "
                "views above."
            ),
            ui.row(
                ui.column(
                    4,
                    ui.input_select(
                        "assoc_stat",
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
            scrollable_plot("assoc_heatmap_plot", 1300, 500),
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

        fig, ax = plt.subplots(figsize=(10, 4.5), layout="tight")
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
                nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), layout="tight"
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

        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="tight")
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
    def assoc_heatmap_plot():
        ds = dataset()
        X = ds.adata
        celltype_col = ds.spec.celltype_col
        stat = input.assoc_stat()
        top_perc = float(input.top_perc())

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
                normalize="global",
                scale=scale,
                show_counts=True,
            )
            ax.set_title(f"{label} Cell Type Association")
        fig.suptitle(f"Cell Type Associations ({stat})", fontweight="bold")
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
        if not dataset().has_pacmap:
            return ui.markdown(
                "Per-cell PaCMAP coordinates are unavailable for this "
                "dataset's factors file. Regenerate it with "
                "`cccrise.regenerate_factors` (or `add_cell_projections`) to "
                "add them."
            )
        celltypes = sorted(
            dataset().adata.obs[dataset().spec.celltype_col].unique().tolist()
        )
        return ui.TagList(
            ui.output_plot("projections_plot", height="500px"),
            ui.hr(),
            ui.h4("Sender/Receiver Strength"),
            ui.markdown(
                "How many cells are a *strong* sender, receiver, both, or "
                'neither for the selected component -- "strong" meaning '
                "the cell's sender (or receiver) weight is at or above the "
                "given percentile across *all* cells in the dataset."
            ),
            ui.row(
                ui.column(
                    6,
                    ui.input_select("tt_celltype", "Cell Type", choices=celltypes),
                ),
                ui.column(
                    6,
                    ui.input_slider(
                        "tt_percentile",
                        "Strength Threshold (percentile)",
                        min=50,
                        max=99,
                        value=90,
                        step=1,
                    ),
                ),
            ),
            ui.markdown("###### Truth Table (Selected Cell Type)"),
            ui.output_data_frame("truth_table"),
            ui.markdown(
                "###### Strength Category PaCMAP (All Cell Types)\n"
                "The same four categories, plotted spatially across every "
                "cell in the dataset rather than counted within one cell "
                "type."
            ),
            scrollable_plot("strength_category_pacmap_plot", 650, 550),
            ui.markdown(
                "###### Sender vs. Receiver Weight (All Cell Types)\n"
                "The raw weights the truth table and PaCMAP above threshold "
                "into strong/not-strong -- threshold-agnostic, so it shows "
                "the full continuous relationship (and how it differs by "
                "cell type) rather than a binary call. Dashed lines mark "
                "the current percentile threshold for reference only."
            ),
            scrollable_plot("sender_receiver_scatter_plot", 900, 700),
        )

    @reactive.calc
    def strength_weights():
        ds = dataset()
        cmp = input.component()
        sender = np.asarray(ds.adata.obsm["sc_B"])[:, cmp - 1]
        receiver = np.asarray(ds.adata.obsm["rc_C"])[:, cmp - 1]
        percentile = input.tt_percentile()
        return (
            sender,
            receiver,
            float(np.percentile(sender, percentile)),
            float(np.percentile(receiver, percentile)),
        )

    @render.data_frame
    def truth_table():
        ds = dataset()
        celltype_col = ds.spec.celltype_col
        sender, receiver, sender_threshold, receiver_threshold = strength_weights()

        in_celltype = ds.adata.obs[celltype_col].to_numpy() == input.tt_celltype()
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
        return render.DataGrid(table)

    @render.plot
    def strength_category_pacmap_plot():
        ds = dataset()
        cmp = input.component()
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

        fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="tight")
        plot_categorical_scatter(
            np.asarray(ds.adata.obsm["PaCMAP"]),
            pd.Series(categories),
            ax,
            color_key=STRENGTH_CATEGORY_COLORS,
        )
        assign_labels(ax)
        ax.set_title(f"Cmp. {cmp} Strength Category")
        return fig

    @render.plot
    def sender_receiver_scatter_plot():
        ds = dataset()
        cmp = input.component()
        celltype_col = ds.spec.celltype_col
        sender, receiver, sender_threshold, receiver_threshold = strength_weights()

        fig, ax = plt.subplots(figsize=(9, 7), layout="tight")
        plot_categorical_scatter(
            np.column_stack([sender, receiver]),
            ds.adata.obs[celltype_col],
            ax,
        )
        ax.axvline(sender_threshold, color="0.3", ls="--", lw=1.0)
        ax.axhline(receiver_threshold, color="0.3", ls="--", lw=1.0)
        ax.set(
            xlabel=f"Cmp. {cmp} Sender Weight",
            ylabel=f"Cmp. {cmp} Receiver Weight",
            title=f"Cmp. {cmp} Sender vs. Receiver Weight by Cell Type",
        )
        return fig

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
        top_perc = input.top_perc()
        return ui.TagList(
            ui.markdown("#### Cell Type Loadings Summary for Selected Component"),
            ui.markdown(
                f"_A cell counts as 'top' here if it's in the sidebar's Top "
                f"Cell % Threshold, currently **{top_perc:g}%**; the mean "
                "loading columns are unaffected by that threshold._"
            ),
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
