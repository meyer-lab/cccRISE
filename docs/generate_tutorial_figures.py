"""Generate the tutorial's figures from the canonical ALAD fit and its
committed rank selection sweeps. Each block mirrors a code block in
`tutorial.rst`; the factorization itself is loaded rather than refit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cccrise.app.plots import (
    pacmap_axes,
    plot_categorical_scatter,
    plot_celltype_association,
    plot_celltype_points,
    plot_cp_factor_heatmaps,
    plot_interaction_heatmap,
    plot_lr_rank,
    plot_value_pacmap,
)
from cccrise.factors import load_factors_to_anndata
from cccrise.figures import common, style
from cccrise.runners.datasets import DATASETS, RANK_SELECTION_DIR

OUTPUT_DIR = Path(__file__).parent / "_static" / "tutorial_images"
DPI = 200

spec = DATASETS["alad"]
COMPONENT = 4  # the 6-month outcome-associated CP component


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style.apply_style()
    rise_rank, cp_rank = common.selected_ranks("alad")

    sweep = common.load_runner_output(spec.rise_rank_filename(), RANK_SELECTION_DIR)
    fig, ax = plt.subplots(figsize=(5, 4))
    common.plot_rise_rank(sweep, ax, rank=rise_rank)
    save(fig, "rise_rank")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cp_sweep = common.production_fits(common.load_cpd_rank("alad"))
    common.plot_cpd_fit_r2x(cp_sweep, ax[0], rank=cp_rank)
    common.plot_cpd_imputation_r2x(common.load_cpd_imputation("alad"), ax[0])
    common.plot_collinearity_boxes(ax[1], "alad", rank=cp_rank)
    fig.tight_layout()
    save(fig, "cpd_rank")

    X = load_factors_to_anndata(spec.factors_path(), adata=spec.load())

    fig, axes = plt.subplots(1, 4, figsize=(16, 8), constrained_layout=True)
    plot_cp_factor_heatmaps(X, axes, condition_col="dsco_id", lr_weight=0.15)
    for ax, title in zip(axes, ["Condition", "Sender", "Receiver", "LR pair"]):
        ax.set_title(f"{title} factor")
    save(fig, "cp_factors")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    points = np.asarray(X.obsm["PaCMAP"])
    plot_celltype_points(axes[0], X, "alad", points)
    outcome = X.obs["6monthcondition"].astype(str).to_numpy()
    colors = dict(zip(sorted(set(outcome)), style.CATEGORICAL, strict=False))
    plot_categorical_scatter(axes[1], points, outcome, colors)
    for ax, title in zip(axes, ["Cell type", "6-month outcome"]):
        pacmap_axes(ax)
        ax.set_title(title)
    save(fig, "pacmap")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for ax, key, role in zip(axes, ["sc_B", "rc_C"], ["Sender", "Receiver"]):
        values = X.obsm[key][:, COMPONENT - 1]
        plot_value_pacmap(X, values, ax, f"Cmp. {COMPONENT} {role} Cells", "Weight", 99)
    save(fig, "component_pacmap")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, role in zip(axes, ["sender", "receiver"]):
        plot_celltype_association(X, ax, role, "broad_cell_type", "mean", 10)
        ax.set_title(f"Mean {role} weight")
    save(fig, "celltype_association")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lr_rank(X, COMPONENT, ax, 10, False, 10, (style.COLORS["emphasis"], "grey"))
    save(fig, "lr_rank")

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_interaction_heatmap(
        X,
        COMPONENT,
        "CD70",
        "CD27",
        ax,
        "CD4 T cells",
        "CD4 T cells",
        "broad_cell_type",
    )
    save(fig, "expression_product")


if __name__ == "__main__":
    main()
