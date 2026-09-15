"""Shiny app for interactively exploring RISE / CCC-RISE decomposition output.

Run with:
    shiny run --reload cccrise/app/app.py
    shiny run --host 0.0.0.0 --port 8000 --reload cccrise/app/app.py
"""

import matplotlib.pyplot as plt
import numpy as np
from shiny import App, reactive, render, ui

from cccrise.figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_interaction_heatmap,
    plot_lr_factors,
    plot_lr_factors_partial,
)
from cccrise.figures.commonFuncs.plotPaCMAP import (
    plot_labels_pacmap,
    plot_wc_pacmap,
    plot_wc_per_celltype,
)
from cccrise.app.data import DATASETS, load_dataset, load_raw_dataset
from cccrise.app.lr_pairs import parse_lr_pairs

DATASET_CHOICES = {key: spec.label for key, spec in DATASETS.items()}

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("dataset", "Dataset", choices=DATASET_CHOICES),
        ui.input_slider("component", "CP Component", min=1, max=1, value=1, step=1),
        ui.input_radio_buttons(
            "weight_source",
            "Cell weight source",
            choices={"sender": "Sender (B)", "receiver": "Receiver (C)"},
            selected="sender",
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
            ui.output_plot("factor_heatmaps", height="650px"),
        ),
        ui.nav_panel(
            "Cell Projections",
            ui.output_ui("projections_body"),
        ),
        ui.nav_panel(
            "Per-Cell-Type Distributions",
            ui.output_ui("distributions_body"),
        ),
        ui.nav_panel(
            "LR Explorer",
            ui.output_ui("lr_explorer_body"),
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
                "it with `cellcommunicationpf2.regenerate_factors`."
            )
        return None

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
    # Per-Cell-Type Distributions
    # ------------------------------------------------------------------
    @render.ui
    def distributions_body():
        if not dataset().has_projections:
            return ui.markdown(
                "Per-cell weights are unavailable for this dataset "
                "(raw single-cell data not found)."
            )
        return ui.output_plot("distributions_plot", height="600px")

    @render.plot
    def distributions_plot():
        ds = dataset()
        X = ds.adata
        cmp = input.component()

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        plot_wc_per_celltype(
            X, cmp, axes[0], cellType=ds.spec.celltype_col, factor_matrix="B"
        )
        axes[0].set_title(f"Cmp. {cmp} Sender Weight by Cell Type")

        plot_wc_per_celltype(
            X, cmp, axes[1], cellType=ds.spec.celltype_col, factor_matrix="C"
        )
        axes[1].set_title(f"Cmp. {cmp} Receiver Weight by Cell Type")

        return fig

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
        top_bottom = ui.output_plot("lr_top_bottom_plot", height="350px")

        raw = load_raw_dataset(input.dataset())
        if raw.error is not None:
            return ui.TagList(top_bottom, ui.markdown(f"⚠️ {raw.error}"))

        celltypes = sorted(raw.adata.obs[ds.spec.celltype_col].unique().tolist())
        pair_labels = [p.label for p in simple_lr_pairs()]

        return ui.TagList(
            top_bottom,
            ui.hr(),
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
    def lr_top_bottom_plot():
        ds = dataset()
        cmp = input.component()
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
        plot_lr_factors_partial(ds.adata, cmp, axes[0], geneAmount=10, top=True)
        axes[0].set_title(f"Cmp. {cmp}: Top LR Pairs")
        plot_lr_factors_partial(ds.adata, cmp, axes[1], geneAmount=10, top=False)
        axes[1].set_title(f"Cmp. {cmp}: Bottom LR Pairs")
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


app = App(app_ui, server)
