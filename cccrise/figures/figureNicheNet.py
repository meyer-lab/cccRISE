"""
Figure: NicheNet-style downstream-target validation of CPD components
(``cccrise.nichenet``), plotted from the cached scores written by
``uv run cccrise-nichenet --dataset <dataset>``.

For every CPD component of the selected dataset's finalized factors, scores
each of its top-loading ligands' curated NicheNet v2 downstream targets for
differential expression in the component's implicated receiver population
(``component_accuracy_scores``, run by the ``run_nichenet`` runner -- this
figure only plots), under both receiver-population definitions:

- "within_celltype": receiver cells restricted to the cell type with the
  highest mean receiver-eigenstate loading for that component.
- "vs_rest": receiver cells drawn from the whole dataset.

Panels, in order: one per-ligand AUROC panel per component for each mode
(left column "within_celltype", right column "vs_rest"), followed by a
per-component summary row (mean AUROC, and how many of each component's top
ligands could actually be scored).

No null/significance comparison is computed here (deferred) -- the vertical
dashed line at AUROC=0.5 marks chance, not a tested threshold.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..runners.datasets import DATASETS
from .common import getSetup, load_runner_csv, subplotLabel

# Change this to switch dataset; must be a key in `runners.datasets.DATASETS`
# with cached scores already produced by `uv run cccrise-nichenet`.
# Only "alad" has a finalized fit today; "ibd" is wired up for when it does.
SELECTED_DATASET = "alad"

EMPHASIS_COLOR = "#3a6b9c"
MUTED_COLOR = "0.72"
VSREST_COLOR = "#c98b3a"
CHANCE_COLOR = "0.4"


def _component_summary(scores: pd.DataFrame, components: list[int]) -> pd.DataFrame:
    summary = (
        scores.groupby("component")["auroc"]
        .agg(mean_auroc="mean", median_auroc="median", n_scored="count")
        .reindex(components)
    )
    summary["n_scored"] = summary["n_scored"].fillna(0).astype(int)
    return summary


def plot_component_ligands(scores: pd.DataFrame, cmp: int, ax, mode_label: str) -> None:
    """Per-ligand AUROC bars for one component/receiver-mode combination."""
    full = scores[scores["component"] == cmp]
    sub = full.dropna(subset=["auroc"]).sort_values("auroc")

    celltype = None
    if len(full) and full["celltype"].notna().any():
        celltype = full["celltype"].iloc[0]

    title_bits = [f"Cmp{cmp}", mode_label]
    if celltype:
        title_bits.append(str(celltype))
    title = " - ".join(title_bits) + f" (n={len(sub)}/{len(full)})"

    if sub.empty:
        ax.text(
            0.5,
            0.5,
            "No scoreable ligands",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    labels = [f"{lig}→{rec}" for lig, rec in zip(sub["ligand"], sub["receptor"])]
    colors = [EMPHASIS_COLOR if a > 0.5 else MUTED_COLOR for a in sub["auroc"]]
    ax.barh(labels, sub["auroc"], color=colors)
    ax.axvline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Ligand-activity AUROC")
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis="y", labelsize=6)

    handles = [
        Patch(facecolor=EMPHASIS_COLOR, label="AUROC > 0.5"),
        Patch(facecolor=MUTED_COLOR, label="AUROC ≤ 0.5"),
        Line2D(
            [0], [0], color=CHANCE_COLOR, linestyle="--", linewidth=0.8, label="Chance"
        ),
    ]
    ax.legend(handles=handles, fontsize=5, loc="lower right")


def plot_component_summary(
    summary_within: pd.DataFrame,
    summary_vsrest: pd.DataFrame,
    components: list[int],
    ax,
    value_col: str,
    ylabel: str,
    title: str,
    chance_line: bool = False,
) -> None:
    """Grouped bar chart comparing the two receiver modes, one bar-pair per
    component."""
    positions = np.arange(len(components))
    width = 0.38

    ax.bar(
        positions - width / 2,
        summary_within[value_col].to_numpy(dtype=float),
        width=width,
        color=EMPHASIS_COLOR,
        label="within_celltype",
    )
    ax.bar(
        positions + width / 2,
        summary_vsrest[value_col].to_numpy(dtype=float),
        width=width,
        color=VSREST_COLOR,
        label="vs_rest",
    )

    if chance_line:
        ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=0.8)
        ax.set_ylim(0, 1)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Cmp{c}" for c in components])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7)
    ax.legend(fontsize=6)


def makeFigure():
    spec = DATASETS[SELECTED_DATASET]
    scores = load_runner_csv(spec.nichenet_scores_filename())
    components = sorted(scores["component"].unique())
    cp_rank = len(components)

    scores_within = scores[scores["receiver_mode"] == "within_celltype"]
    scores_vsrest = scores[scores["receiver_mode"] == "vs_rest"]

    summary_within = _component_summary(scores_within, components)
    summary_vsrest = _component_summary(scores_vsrest, components)

    nrows = cp_rank + 1
    ax, f = getSetup((11, 3.0 * nrows), (nrows, 2))
    subplotLabel(ax)

    for i, cmp in enumerate(components):
        plot_component_ligands(scores_within, cmp, ax[i * 2], "within_celltype")
        plot_component_ligands(scores_vsrest, cmp, ax[i * 2 + 1], "vs_rest")

    summary_row = cp_rank * 2
    plot_component_summary(
        summary_within,
        summary_vsrest,
        components,
        ax[summary_row],
        value_col="mean_auroc",
        ylabel="Mean ligand-activity AUROC",
        title=f"Per-component mean AUROC ({SELECTED_DATASET})",
        chance_line=True,
    )
    plot_component_summary(
        summary_within,
        summary_vsrest,
        components,
        ax[summary_row + 1],
        value_col="n_scored",
        ylabel="Ligands scored (of top-loading set)",
        title=f"Per-component ligand coverage ({SELECTED_DATASET})",
    )

    print(f"=== NicheNet component summary: {SELECTED_DATASET}, within_celltype ===")
    print(summary_within.to_string())
    print(f"=== NicheNet component summary: {SELECTED_DATASET}, vs_rest ===")
    print(summary_vsrest.to_string())

    return f
