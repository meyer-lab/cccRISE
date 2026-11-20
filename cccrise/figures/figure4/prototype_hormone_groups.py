"""PROTOTYPE: Figure 2's label-free sender/receiver split (`figure2._common.
split_cells`) applied to the IBD hormone programs C3 and C4, one row per
component laid out as Figure 2's: every cell's sender vs receiver weight with
the groups; a dot plot of the candidate gut markers that stratify sender from
receiver (Wilcoxon, BH over the candidates, as `figure2.epithelial_markers`)
plus the top LR pairs' genes; and each group's annotated composition.

    uv run python -m cccrise.figures.figure4.prototype_hormone_groups

The per-cell table is cached in `OUTPUT` (delete it to recompute)."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

from ...runners.datasets import DATASETS
from ...validation.nichenet import top_pairs
from .. import common, style
from ..figure2._common import GROUP_COLORS, GROUPS, plot_split_boundary, split_cells
from ..figure2.epithelial_markers import MIN_DETECTED_DIFFERENCE
from . import _common as S

COMPONENTS = (3, 4)
OUTPUT = common.OUTPUT_DIR / "prototype_hormone_groups"
CELLS = OUTPUT / "cells.parquet"
#: Candidate ileal epithelial markers, tested sender vs receiver per component
CANDIDATE_MARKERS = {
    "EE hormones": ["GCG", "PYY", "INSL5", "CCK", "SCT", "NTS", "GIP", "SST", "GHRL"],
    "Pan-EE": ["CHGA", "CHGB", "SCGN", "NEUROD1"],
    "EC": ["TPH1", "TAC1", "FEV", "LMX1A"],
    "Tuft": ["POU2F3", "DCLK1", "TRPM5", "AVIL", "LRMP", "SH2D6"],
    "Goblet": ["MUC2", "TFF3", "SPINK4"],
    "Enterocyte": ["ALDOB", "APOA4", "FABP2", "RBP2"],
    "Stem": ["LGR5", "OLFM4"],
}
#: Annotated composition categories: the fine types named here, else lineage
COMPOSITION = {
    "EE": style.CATEGORICAL[3],
    "EC": style.CATEGORICAL[0],
    "Tuft": style.CATEGORICAL[2],
    "Goblet": style.CATEGORICAL[4],
    "Enterocyte": style.CATEGORICAL[5],
    "Stem cell": style.CATEGORICAL[8],
    "Immune": "0.8",
}
FINE_CATEGORY = {
    "Enteroendocrine cells": "EE",
    "Enterochromaffin cells": "EC",
    "Tuft cells": "Tuft",
}


def main() -> None:
    style.apply_style()
    cells, pair_genes = cell_table()
    rows = [panels(cells, k, pair_genes[k]) for k in COMPONENTS]
    fig = common.compose_fixed(rows, width=style.FIGURE_WIDTH_DOUBLE)
    fig.savefig(OUTPUT / "prototype_hormone_groups.png", dpi=300)
    pd.concat(
        [_stratifying_markers(cells, k)[1].assign(Component=k) for k in COMPONENTS]
    ).to_csv(OUTPUT / "marker_tests.csv", index=False)


def cell_table() -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """Every cell's composition category, C3/C4 weights and group, and the
    expression of the candidate markers and top pair genes; and each
    component's top pair genes."""
    lr = S.factors()["lr_pair"].values.astype(str)
    D = S.factors()["cp_D"].values
    pair_genes = {k: _top_pair_genes(D, lr, k) for k in COMPONENTS}
    if CELLS.exists():
        return pd.read_parquet(CELLS), pair_genes
    adata = common.get_cell_projections(S.DATASET)
    spec = DATASETS[S.DATASET]
    genes = [g for genes in CANDIDATE_MARKERS.values() for g in genes]
    genes += [g for k in COMPONENTS for g in pair_genes[k]]
    cells = adata[:, [g for g in dict.fromkeys(genes) if g in adata.var_names]].to_df()
    obs = cast(pd.DataFrame, adata.obs)
    cells["Composition"] = _composition_labels(
        obs[spec.fine_celltype_column].astype(str),
        obs[spec.celltype_column].astype(str),
    )
    for k in COMPONENTS:
        sender, receiver = adata.obsm["sc_B"][:, k - 1], adata.obsm["rc_C"][:, k - 1]
        cells[f"C{k} sender"], cells[f"C{k} receiver"] = sender, receiver
        cells[f"C{k} group"] = _group_labels(sender, receiver)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells.reset_index(drop=True).to_parquet(CELLS)
    return cells.reset_index(drop=True), pair_genes


def _group_labels(sender: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    labels = np.full(len(sender), "rest", dtype=object)
    for g, mask in split_cells(sender, receiver).items():
        labels[mask] = g
    return labels


def panels(cells: pd.DataFrame, k: int, pair_genes: list[str]) -> list:
    """Figure 2's group row for component `k`: scatter, then dot plot over
    composition."""
    sections = _stratifying_markers(cells, k)[0] | {"Top LR pairs": pair_genes}
    labels = cells[f"C{k} group"]

    def scatter(axs):
        _scatter(
            axs[0],
            cells[f"C{k} sender"].to_numpy(),
            cells[f"C{k} receiver"].to_numpy(),
            labels.to_numpy(),
            k,
        )

    def dots(axs):
        genes = list(dict.fromkeys(g for section in sections.values() for g in section))
        mean, detected = S.group_stats(cells[genes], labels, list(GROUPS))
        for frame in (mean, detected):
            frame.index = [g.capitalize() for g in frame.index]
        S.dotplot(axs[0], mean, detected, sections)

    def composition(axs):
        _composition(axs[0], cells["Composition"].to_numpy(), labels.to_numpy())

    def panel(name, axes, plot):
        return SimpleNamespace(__name__=f"c{k}_{name}", AXES=(axes,), plot=plot)

    return [
        panel("scatter", "square", scatter),
        [panel("dots", "strip", dots), panel("composition", "short", composition)],
    ]


def _composition_labels(fine: pd.Series, lineage: pd.Series) -> np.ndarray:
    labels = fine.map(FINE_CATEGORY)
    labels = labels.fillna(fine.str.startswith("Goblet").map({True: "Goblet"}))
    labels = labels.fillna(lineage.where(lineage.isin(["Enterocyte", "Stem cell"])))
    return labels.fillna("Immune").to_numpy()


def _top_pair_genes(D, lr: np.ndarray, k: int) -> list[str]:
    """Ligand then receptor genes of component `k`'s top LR pairs (NicheNet's
    relative-loading rule, as Figure 2)."""
    ligands = pd.Series(lr).str.split("^").str[0]
    rows = top_pairs(D, pd.DataFrame({"ligand": ligands}), k, set(ligands))
    names = pd.Series(lr[rows]).str.split("^")
    genes = [g for side in (0, 1) for g in names.str[side].str.split("&").explode()]
    return list(dict.fromkeys(genes))


def _stratifying_markers(cells: pd.DataFrame, k: int):
    """({section: candidate markers whose sender vs receiver Wilcoxon q <
    SIG_ALPHA and detected fractions differ by MIN_DETECTED_DIFFERENCE}, every
    candidate's test)."""
    candidates = [
        g for genes in CANDIDATE_MARKERS.values() for g in genes if g in cells
    ]
    labels = cells[f"C{k} group"]
    send = cells.loc[labels == "sender", candidates]
    recv = cells.loc[labels == "receiver", candidates]
    p = np.array([mannwhitneyu(send[g], recv[g]).pvalue for g in candidates])
    tested = pd.DataFrame(
        {
            "Gene": candidates,
            "Wilcoxon p": p,
            "BH q": false_discovery_control(p),
            "Detected (sender)": (send > 0).mean().to_numpy(),
            "Detected (receiver)": (recv > 0).mean().to_numpy(),
        }
    )
    tested["Shown"] = (tested["BH q"] < common.SIG_ALPHA) & (
        (tested["Detected (sender)"] - tested["Detected (receiver)"]).abs()
        >= MIN_DETECTED_DIFFERENCE
    )
    kept = set(tested.loc[tested["Shown"], "Gene"])
    sections = {
        name: [g for g in genes if g in kept]
        for name, genes in CANDIDATE_MARKERS.items()
    }
    return {name: genes for name, genes in sections.items() if genes}, tested


def _scatter(ax, sender, receiver, labels, k: int) -> None:
    rest = labels == "rest"
    ax.scatter(
        sender[rest],
        receiver[rest],
        c=style.COLORS["muted"],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
        label=f"Rest (n={rest.sum():,})",
    )
    for g, color in GROUP_COLORS.items():
        mask = labels == g
        ax.scatter(
            sender[mask],
            receiver[mask],
            c=color,
            s=style.POINT_SIZE_BACKGROUND,
            linewidths=0,
            rasterized=True,
            label=f"{g.capitalize()} (n={mask.sum():,})",
        )
    plot_split_boundary(ax, sender.max(), receiver.max())
    ax.set(xlabel=f"C{k} sender weight", ylabel=f"C{k} receiver weight")
    legend = common.legend(
        ax,
        "below",
        ncols=2,
        columnspacing=0.8,
        handletextpad=0.2,
        fontsize=style.FONT_SIZE_DENSE,
    )
    for handle in legend.legend_handles:
        handle.set_sizes([12])


def _composition(ax, composition: np.ndarray, labels: np.ndarray) -> None:
    fractions = pd.crosstab(labels, composition, normalize="index").reindex(
        index=list(GROUPS), columns=list(COMPOSITION), fill_value=0
    )
    left = np.zeros(len(GROUPS))
    for name, color in COMPOSITION.items():
        ax.barh(
            range(len(GROUPS)),
            fractions[name],
            left=left,
            color=color,
            height=0.7,
            label=name,
        )
        left += fractions[name].to_numpy()
    ax.set_yticks(range(len(GROUPS)), [g.capitalize() for g in GROUPS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells")
    common.legend(ax, "right", ncols=2)


if __name__ == "__main__":
    main()
