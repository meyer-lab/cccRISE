"""
Figure: NicheNet-style downstream-target validation of CPD components
(``cccrise.validation.nichenet``), plotted from the cached scores written by
``uv run cccrise-nichenet --dataset <dataset>``.

For every CPD component of the selected dataset's finalized factors, scores
each of its top-loading ligands' curated NicheNet v2 downstream targets for
differential expression in the component's implicated receiver population
(``component_accuracy_scores``, run by the ``run_nichenet`` runner -- this
figure only plots), under both receiver-population definitions:

- "within_celltype": receiver cells restricted to the cell type(s) with
  comparably high mean receiver-eigenstate loading for that component (see
  ``nichenet.receiver_celltypes_for_component``) -- one or several cell
  types may be pooled together.
- "vs_rest": receiver cells drawn from the whole dataset.

Significance is a permutation test (``nichenet.permutation_null_aurocs``):
each ligand's AUROC is compared to its empirical null distribution from
random relabelings of the same receiver population. Bars are colored by
significance (p < 0.05) and direction (AUROC above/below chance).

Panels, in order: one per-ligand AUROC panel per component for each mode
(left column "within_celltype", right column "vs_rest"), followed by two
summary rows (mean AUROC, ligand coverage, and enriched and depleted
significant-ligand counts; each ligand is counted once per component).
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..runners.datasets import DATASETS
from ..validation.nichenet import summarize_component_scores
from .common import getSetup, load_runner_csv, subplotLabel

# Change this to switch dataset; must be a key in `runners.datasets.DATASETS`
# with cached scores already produced by `uv run cccrise-nichenet`.
SELECTED_DATASET = "alad"

SIG_ALPHA = 0.05
SIG_UP_COLOR = "#3a6b9c"
SIG_DOWN_COLOR = "#b3382c"
NOT_SIG_COLOR = "0.72"
VSREST_COLOR = "#c98b3a"
CHANCE_COLOR = "0.4"


def _significance_color(auroc: float, p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value) or p_value >= SIG_ALPHA:
        return NOT_SIG_COLOR
    return SIG_UP_COLOR if auroc > 0.5 else SIG_DOWN_COLOR


def _component_summary(scores: pd.DataFrame, components: list[int]) -> pd.DataFrame:
    summary = summarize_component_scores(scores, alpha=SIG_ALPHA).reindex(components)
    for column in ("n_scored", "n_enriched", "n_depleted"):
        summary[column] = summary[column].fillna(0).astype(int)
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

    labels = [
        f"{lig}→{rec}" + (" *" if p < SIG_ALPHA else "")
        for lig, rec, p in zip(sub["ligand"], sub["receptor"], sub["p_value"])
    ]
    colors = [_significance_color(a, p) for a, p in zip(sub["auroc"], sub["p_value"])]
    ax.barh(labels, sub["auroc"], color=colors)
    ax.axvline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Ligand-activity AUROC")
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis="y", labelsize=6)

    handles = [
        Patch(facecolor=SIG_UP_COLOR, label=f"p<{SIG_ALPHA}, AUROC>0.5"),
        Patch(facecolor=SIG_DOWN_COLOR, label=f"p<{SIG_ALPHA}, AUROC<0.5"),
        Patch(facecolor=NOT_SIG_COLOR, label="not significant"),
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
        color=SIG_UP_COLOR,
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

    nrows = cp_rank + 2
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
    plot_component_summary(
        summary_within,
        summary_vsrest,
        components,
        ax[summary_row + 2],
        value_col="n_enriched",
        ylabel=f"Ligands enriched (p<{SIG_ALPHA}, AUROC>0.5)",
        title=f"Per-component enriched targets ({SELECTED_DATASET})",
    )
    plot_component_summary(
        summary_within,
        summary_vsrest,
        components,
        ax[summary_row + 3],
        value_col="n_depleted",
        ylabel=f"Ligands depleted (p<{SIG_ALPHA}, AUROC<0.5)",
        title=f"Per-component depleted targets ({SELECTED_DATASET})",
    )

    print(f"=== NicheNet component summary: {SELECTED_DATASET}, within_celltype ===")
    print(summary_within.to_string())
    print(f"=== NicheNet component summary: {SELECTED_DATASET}, vs_rest ===")
    print(summary_vsrest.to_string())

    return f
