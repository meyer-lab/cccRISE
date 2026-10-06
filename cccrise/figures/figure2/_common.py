"""The granularity component shared by Figure 2's panels and its cmp. 5
supplements: the CP component that splits a cell type into sender and
receiver subpopulations, the label-free sender and receiver groups its
per-cell weights define, and their gene expression."""

import functools

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from ...validation.nichenet import top_pairs
from .. import common, style

# The CP component that splits a cell type into sender and receiver
# subpopulations, and the RISE eigenstates of its largest sender (B) and
# receiver (C) loadings
GRANULARITY_DATASET = "alad"
GRANULARITY_COMPONENT = 5
GRANULARITY_SENDER_EIGENSTATE, GRANULARITY_RECEIVER_EIGENSTATE = 17, 15
# Cells are split on their per-cell sender (`obsm["sc_B"]`) and receiver
# (`obsm["rc_C"]`) weights on the component alone (`split_cells`), each
# cutoff a fraction GROUP_TAU of that weight's largest value, then compared by
# gene expression
GROUP_TAU = 0.3
GROUP_COLORS = {"sender": style.COLORS["sender"], "receiver": style.COLORS["receiver"]}
GROUPS = (*GROUP_COLORS, "rest")
DE_GROUPS = tuple(GROUP_COLORS)  # each tested against "rest"
DOT_SIZE_MAX = 30  # pt^2, a gene detected in every cell of a group
DOT_FRACTIONS = (0.25, 0.5, 1.0)  # size legend entries
# Dot plot section names wider than this fraction of their genes' span are
# abbreviated (`label_sections`), keeping neighbors apart
SECTION_FILL = 0.9
SECTION_ABBREVIATIONS = {
    "Macrophages": "Mac.",
    "Monocytes": "Mono.",
    "Dendritic Cells": "DC",
    "CD4 T cells": "CD4 T",
    "CD8 T cells": "CD8 T",
    "NK cells": "NK",
    "Epithelial cells": "Epithelial",
    "Proliferating cells": "Prolif.",
    "Pan-epithelial": "Pan-epi.",
    "Secretory": "Secr.",
    "Top LR pairs": "LR pairs",
}
# Canonical markers of the annotated broad cell types, from the human lung
# cell atlas (Travaglini et al. 2020, Nature), chosen to be specific within
# these eight types (e.g. no pan-T genes shared by the CD4 and CD8 sets).
# Only the gene names come from the literature; the annotations themselves
# are not used.
CELLTYPE_MARKERS = {
    "Macrophages": ["MARCO", "FABP4", "C1QB", "CD68"],
    "Monocytes": ["CD14", "FCN1", "VCAN", "S100A8"],
    "Dendritic Cells": ["CD1C", "FCER1A", "CLEC10A"],
    "CD4 T cells": ["CD4", "IL7R", "CD40LG"],
    "CD8 T cells": ["CD8A", "CD8B", "GZMK"],
    "NK cells": ["KLRF1", "NCAM1", "GNLY"],
    "Epithelial cells": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
    "Proliferating cells": ["MKI67", "TOP2A"],
}


def component_weights() -> tuple[np.ndarray, np.ndarray]:
    """Per-cell (sender, receiver) weight on the granularity component."""
    adata = common.get_cell_projections(GRANULARITY_DATASET)
    return (
        adata.obsm["sc_B"][:, GRANULARITY_COMPONENT - 1],
        adata.obsm["rc_C"][:, GRANULARITY_COMPONENT - 1],
    )


def split_cells(
    sender: np.ndarray, receiver: np.ndarray, tau: float = GROUP_TAU
) -> dict[str, np.ndarray]:
    """{group: cell mask}: the cells passing the sender cutoff (`tau` x the
    largest sender weight) or the receiver cutoff, each assigned to the role
    of its larger weight relative to that weight's maximum."""
    relative_sender, relative_receiver = (
        sender / sender.max(),
        receiver / receiver.max(),
    )
    high = (relative_sender >= tau) | (relative_receiver >= tau)
    sends = relative_sender > relative_receiver
    return {"sender": high & sends, "receiver": high & ~sends}


def top_pair_genes() -> list[str]:
    """Ligand then receptor genes (complexes split) of the component's top LR
    pairs: those `cp_factor_matrices` labels (NicheNet's relative-loading rule
    over every pair)."""
    adata = common.get_cell_projections(GRANULARITY_DATASET)
    names = pd.Series(np.asarray(adata.uns["lr_pairs"]).astype(str)).str.split("^")
    lr_pairs = pd.DataFrame({"ligand": names.str[0]})
    rows = top_pairs(
        adata.uns["D"], lr_pairs, GRANULARITY_COMPONENT, set(lr_pairs["ligand"])
    )
    top = names[rows]
    return present(g for side in (0, 1) for g in top.str[side].str.split("&").explode())


@functools.cache
def cell_groups() -> pd.Series:
    """Each cell's group at `GROUP_TAU`, "rest" if in none."""
    groups = split_cells(*component_weights())
    labels = np.full(len(groups["sender"]), "rest", dtype=object)
    for group, mask in groups.items():
        labels[mask] = group
    return pd.Series(pd.Categorical(labels, categories=GROUPS))


@functools.cache
def rank_genes(groups: tuple[str, ...], reference: str) -> dict[str, pd.DataFrame]:
    """{group: Wilcoxon score and BH q per gene, highest score first} of each
    of `groups` against `reference`. Only the scores and q are used: scanpy's
    fold changes assume natural-log data."""
    adata = common.get_cell_projections(GRANULARITY_DATASET)
    test = anndata.AnnData(
        X=adata.X,
        obs=pd.DataFrame({"group": cell_groups().to_numpy()}, index=adata.obs_names),
        var=pd.DataFrame(index=adata.var_names),
    )
    sc.tl.rank_genes_groups(
        test, "group", groups=list(groups), reference=reference, method="wilcoxon"
    )
    result = test.uns["rank_genes_groups"]
    return {
        g: pd.DataFrame(
            {"score": result["scores"][g], "q": result["pvals_adj"][g]},
            index=result["names"][g],
        )
        for g in groups
    }


def group_expression(genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fraction of cells detecting, mean expression) of `genes` (rows) in
    each group (columns)."""
    adata = common.get_cell_projections(GRANULARITY_DATASET)
    X = adata[:, genes].to_df().to_numpy()
    groups = cell_groups().to_numpy()
    detected, mean = (
        pd.DataFrame(
            {g: stat(X[groups == g]) for g in GROUPS}, index=genes, dtype=float
        )
        for stat in (lambda x: (x > 0).mean(axis=0), lambda x: x.mean(axis=0))
    )
    return detected, mean


def present(genes) -> list[str]:
    """`genes` in the fit's gene set, in order, without repeats."""
    var_names = set(common.get_cell_projections(GRANULARITY_DATASET).var_names)
    return [g for g in dict.fromkeys(genes) if g in var_names]


def plot_dotplot(
    ax,
    sections: dict[str, list[str]],
    show_legend: bool = False,
    fontsize: float | None = None,
):
    """Groups (rows) x genes (columns, under their `sections` names): dot
    size the fraction of the group's cells detecting the gene, color its mean
    expression relative to the gene's largest group mean, every label
    `fontsize` (default `FACTOR_LABEL_FONT_SIZE`). With `show_legend`, the colorbar and size legend beside it.
    Returns the dots."""
    fontsize = fontsize or common.FACTOR_LABEL_FONT_SIZE
    columns = label_sections(ax, sections, fontsize)
    detected, mean = group_expression(list(columns))
    relative = mean / mean.max(axis=1).to_numpy()[:, None]
    x, y = np.meshgrid(list(columns.values()), np.arange(len(GROUPS)), indexing="ij")
    points = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=DOT_SIZE_MAX * detected.to_numpy().ravel(),
        c=relative.to_numpy().ravel(),
        cmap=style.CMAPS["weight"],
        vmin=0,
        vmax=1,
        linewidths=0,
    )
    ax.set_yticks(
        range(len(GROUPS)), [g.capitalize() for g in GROUPS], fontsize=fontsize
    )
    ax.set_ylim(len(GROUPS) - 0.5, -0.5)
    if show_legend:
        common.colorbar(ax, points, "Mean expr.\n/ max group")
        common.legend(
            ax,
            "right",
            handles=[
                ax.scatter([], [], s=DOT_SIZE_MAX * f, color=style.COLORS["ink"])
                for f in DOT_FRACTIONS
            ],
            labels=[f"{f:.0%}" for f in DOT_FRACTIONS],
            title="Detected",
        )
    return points


def plot_markers(ax, markers: dict[str, list[str]], **kwargs):
    """`plot_dotplot` of literature `markers` ({section: genes}), keeping the
    genes in the fit's gene set."""
    return plot_dotplot(
        ax, {name: present(genes) for name, genes in markers.items()}, **kwargs
    )


def label_sections(
    ax, sections: dict[str, list[str]], fontsize: float
) -> dict[str, int]:
    """Gene tick labels along x at 45 degrees, each section's name above its
    genes (its `SECTION_ABBREVIATIONS` entry if the name would overrun them)
    and thin lines between sections, all `fontsize`; returns {gene: column}."""
    genes = [g for section in sections.values() for g in section]
    ax.set_xticks(
        range(len(genes)),
        genes,
        fontsize=fontsize,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.5, len(genes) - 0.5)
    ax.tick_params(length=0)
    renderer = ax.figure.canvas.get_renderer()
    column_width = ax.bbox.width / len(genes)
    columns, start = {}, 0
    for name, section in sections.items():
        end = start + len(section)
        text = ax.text(
            (start + end - 1) / 2,
            1.03,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )
        if (
            text.get_window_extent(renderer).width
            > SECTION_FILL * len(section) * column_width
        ):
            text.set_text(SECTION_ABBREVIATIONS.get(name, name))
        if end < len(genes):
            ax.axvline(end - 0.5, color=style.COLORS["muted"], lw=style.LINE_WIDTH_THIN)
        columns |= {gene: start + i for i, gene in enumerate(section)}
        start = end
    return columns


def plot_weight_scatter(
    ax,
    groups: dict[str, np.ndarray],
    colors: dict[str, str],
    rest_label: str | None = None,
):
    """Per-cell component sender vs. receiver weight, the cells of each of
    `groups` ({legend label: mask}) colored `colors[label]`, the rest grey
    (in the legend as `rest_label`, if given)."""
    sender, receiver = component_weights()
    rest = ~np.logical_or.reduce(list(groups.values()))
    ax.scatter(
        sender[rest],
        receiver[rest],
        c=style.COLORS["muted"],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
        label=f"{rest_label} (n={rest.sum()})" if rest_label else None,
    )
    for label, mask in groups.items():
        ax.scatter(
            sender[mask],
            receiver[mask],
            c=colors[label],
            s=style.POINT_SIZE_BACKGROUND,
            linewidths=0,
            rasterized=True,
            label=f"{label} (n={mask.sum()})",
        )
    ax.set(
        xlabel=f"Cmp. {GRANULARITY_COMPONENT} sender weight",
        ylabel=f"Cmp. {GRANULARITY_COMPONENT} receiver weight",
    )
