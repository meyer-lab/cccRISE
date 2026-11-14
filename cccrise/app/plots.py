"""Explorer-only plots: interactive views the manuscript figures fix to one
statistic, component, or cell type (or omit), so the app keeps its own."""

import math
from typing import cast

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from cccrise.figures import common, style
from cccrise.figures.figure3._common import plot_ranked_loadings
from cccrise.runners.datasets import DATASETS
from cccrise.utils import (
    binned_complex_ligand_product_matrix,
    binned_expression_product_matrix,
    celltype_component_association,
)

FACTOR_KEYS = {"sender": "sc_B", "receiver": "rc_C"}

# =============================================================================
# CP factor matrices (Figure 2's heatmaps, with every condition labeled and
# the LR pairs trimmed by a weight threshold)
# =============================================================================


def plot_cp_factor_heatmaps(
    adata: anndata.AnnData, axes, condition_col: str, lr_weight: float
) -> None:
    """Condition (A), sender (B) and receiver (C) eigenstate, and LR pair (D)
    loadings, keeping only LR pairs whose largest |loading| exceeds
    `lr_weight`."""
    conditions = (
        cast(pd.DataFrame, adata.obs)
        .drop_duplicates("condition_unique_idxs")
        .set_index("condition_unique_idxs")[condition_col]
        .sort_index()
        .to_numpy()
    )
    D = np.asarray(adata.uns["D"], dtype=float)
    kept = np.abs(D).max(axis=1) > lr_weight
    rows = {
        "A": (np.asarray(adata.uns["A"], dtype=float), conditions, True),
        "B": (np.asarray(adata.uns["B"], dtype=float), None, False),
        "C": (np.asarray(adata.uns["C"], dtype=float), None, False),
        "D": (D[kept], np.asarray(adata.uns["lr_pairs"])[kept], True),
    }
    for ax, (loadings, names, ordered) in zip(axes, rows.values(), strict=True):
        order = common.dominant_order(loadings) if ordered else np.arange(len(loadings))
        image = common.plot_factor_heatmap(ax, loadings[order], "CP component")
        names = np.arange(1, len(loadings) + 1) if names is None else names[order]
        ax.set_yticks(range(len(names)), names)
    # Lenient LR thresholds keep enough pairs for their labels to collide
    axes[-1].tick_params(
        axis="y", labelsize=float(np.clip(200 / max(kept.sum(), 1), 4, 9))
    )
    plt.colorbar(image, ax=axes[-1], label="Loading / max |loading|")


# =============================================================================
# Cell type association heatmaps (Figure 3's sender/receiver and the
# eigenstate supplement's, with a selectable statistic)
# =============================================================================

STAT_LABELS = {
    "mean": "Mean Cell Loading",
    "effect_size": "Cohen's $d$ vs. Other Cells",
    "enrichment": "log$_2$ Enrichment in Top {top_perc:g}% of Cells",
    "pct_top": "% of Cells in Top {top_perc:g}%",
}


def plot_celltype_association(
    adata: anndata.AnnData,
    ax: plt.Axes,
    factor_type: str,
    celltype_col: str,
    stat: str,
    top_perc: float,
    scale: float | None = None,
    cbar: bool = True,
) -> None:
    """Cell type x component heatmap of `celltype_component_association`'s
    `stat`, divided by `scale` (default: its own largest magnitude); pass one
    `scale` to several panels to share their color scale."""
    df = celltype_component_association(
        adata,
        factor_type=factor_type,
        celltype_col=celltype_col,
        stat=stat,
        top_perc=top_perc,
    )
    values = df.to_numpy(dtype=float)
    values = values / (float(np.nanmax(np.abs(values))) if scale is None else scale)
    counts = adata.obs[celltype_col].astype(str).value_counts()
    sns.heatmap(
        data=values,
        xticklabels=df.columns.to_numpy(),
        yticklabels=[f"{name} (n={counts[name]})" for name in df.index],
        ax=ax,
        center=0,
        cmap=style.CMAPS["diverging"],
        vmin=-1,
        vmax=1,
        cbar=cbar,
        cbar_kws={
            "label": f"{STAT_LABELS[stat].format(top_perc=top_perc)} (Normalized)"
        },
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel=df.columns.name, ylabel="")


# =============================================================================
# LR pair rank curve (Figure 3's, with selectable highlighting and callouts)
# =============================================================================


def plot_lr_rank(
    adata: anndata.AnnData,
    cmp: int,
    ax: plt.Axes,
    highlight: int | str,
    include_negative: bool,
    max_labels: int,
    colors: tuple[str, str],
) -> None:
    """Every LR pair's `cmp` loading against its rank, the `highlight`ed pairs
    (top N, or above Q3 / Tukey's upper fence; mirrored on the negative tail
    with `include_negative`) marked in `colors` (top, negative), and up to
    `max_labels` per tail named in a callout column."""
    weights = np.asarray(adata.uns["D"], dtype=float)[:, cmp - 1]
    plot_ranked_loadings(ax, weights, 0)
    order = np.argsort(-weights)
    ranked = pd.DataFrame(
        {
            "lr_pair": np.asarray(adata.uns["lr_pairs"]).astype(str)[order],
            "weight": weights[order],
            "rank": np.arange(1, weights.size + 1),
        }
    )
    top = _highlight_mask(ranked["weight"].to_numpy(), highlight)
    bottom = np.zeros_like(top)
    if include_negative:
        # Mirror the rule by ranking the flipped weights, then flipping back
        bottom = _highlight_mask(-ranked["weight"].to_numpy()[::-1], highlight)[::-1]
    for mask, color, from_top in ((top, colors[0], True), (bottom, colors[1], False)):
        selected = ranked[mask]
        if selected.empty:
            continue
        ax.scatter(
            selected["rank"],
            selected["weight"],
            s=20,
            color=color,
            zorder=3,
            clip_on=False,
        )
        labeled = selected.head(max_labels) if from_top else selected.tail(max_labels)
        _annotate_callouts(ax, selected, labeled, color, from_top)
    ax.set(
        xlabel="LR Pair Rank",
        ylabel=f"Cmp. {cmp} Weight",
        title=f"Cmp. {cmp} Ligand-Receptor Pair Weights by Rank",
    )


def _highlight_mask(weights: np.ndarray, highlight: int | str) -> np.ndarray:
    """Mask over descending `weights` of the pairs to mark."""
    if isinstance(highlight, int):
        return np.arange(weights.size) < highlight
    q1, q3 = np.percentile(weights, [25, 75])
    threshold = {"tukey": q3 + 1.5 * (q3 - q1), "q3": q3}[highlight]
    return weights > threshold


def _annotate_callouts(
    ax: plt.Axes, selected: pd.DataFrame, labeled: pd.DataFrame, color, from_top: bool
) -> None:
    """Label marked points in a column just beyond the last marked rank, each
    label at least a label height from the previous one (a sparse factor's
    head packs its weights too tightly to label in place), extending the
    axis if the column overruns it."""
    x0, x1 = ax.get_xlim()
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    labeled = labeled if from_top else labeled.iloc[::-1]
    positions = _stack_labels(labeled["weight"].to_numpy(), gap, from_top)
    if from_top and positions[-1] < lo:
        ax.set_ylim(bottom=positions[-1] - 0.04 * (hi - lo))
    elif not from_top and positions[-1] > hi:
        ax.set_ylim(top=positions[-1] + 0.04 * (hi - lo))

    offset = 0.03 * (x1 - x0)
    label_x = (
        selected["rank"].max() + offset if from_top else selected["rank"].min() - offset
    )
    for (name, weight, rank), position in zip(
        labeled[["lr_pair", "weight", "rank"]].itertuples(index=False),
        positions,
        strict=True,
    ):
        ax.annotate(
            f"{name} ({weight:+.3f})",
            xy=(rank, weight),
            xytext=(label_x, position),
            fontsize=9,
            va="center",
            ha="left" if from_top else "right",
            color=color,
            # So the flat middle of the curve does not strike through a label
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.55"},
        )


def _stack_labels(weights: np.ndarray, gap: float, from_top: bool) -> np.ndarray:
    """Label heights at `weights`, each pushed at least `gap` beyond the
    previous one: downward from the top tail, upward from the negative."""
    sign = 1 if from_top else -1
    positions = sign * weights.astype(float)
    for i in range(1, len(positions)):
        positions[i] = min(positions[i], positions[i - 1] - gap)
    return sign * positions


# =============================================================================
# Per-cell weight distributions by cell type
# =============================================================================


def celltype_grid_shape(
    adata: anndata.AnnData, celltype_col: str, ncols: int = 4
) -> tuple[int, int]:
    """Rows and columns giving every cell type its own panel."""
    n_celltypes = adata.obs[celltype_col].nunique()
    ncols = min(ncols, n_celltypes)
    return math.ceil(n_celltypes / ncols), ncols


def plot_weight_histogram_grid(
    adata: anndata.AnnData,
    cmp: int,
    axes,
    celltype_col: str,
    top_perc: float,
    log_y: bool,
    colors: tuple[str, str],
) -> None:
    """One panel per cell type of its cells' `cmp` sender and receiver weight
    histograms (as fractions of its cells), with bins, limits, and the dashed
    top-`top_perc`% thresholds taken from all cells so panels compare. Shows
    the shapes behind the association heatmap's summaries, e.g. a bimodal
    population whose halves cancel to a mean near zero."""
    labels = adata.obs[celltype_col].astype(str).to_numpy()
    weights = {
        role: np.asarray(adata.obsm[key], dtype=float)[:, cmp - 1]
        for role, key in FACTOR_KEYS.items()
    }
    lo = min(float(w.min()) for w in weights.values())
    hi = max(float(w.max()) for w in weights.values())
    edges = np.linspace(lo, hi, 51).tolist()
    axes = np.asarray(axes).ravel()
    for ax, celltype in zip(axes, np.unique(labels), strict=False):
        mask = labels == celltype
        ax.axvline(0.0, color="0.6", lw=0.6, zorder=0)
        for (role, values), color in zip(weights.items(), colors, strict=True):
            subset = values[mask]
            ax.hist(
                subset,
                bins=edges,
                weights=np.full(subset.size, 1 / subset.size),
                histtype="stepfilled",
                alpha=0.45,
                color=color,
                edgecolor=color,
                lw=0.8,
                label=role.capitalize(),
            )
            ax.axvline(
                np.percentile(values, 100 - top_perc), color=color, ls="--", lw=0.8
            )
        if log_y:
            ax.set_yscale("log")
        # Headroom so the legend clears the tallest bin
        ax.set_ylim(top=ax.get_ylim()[1] * (8 if log_y else 1.6))
        ax.set_xlim(lo, hi)
        ax.set(xlabel=f"Cmp. {cmp} Weight", ylabel="Fraction of Cells")
        ax.set_title(f"{celltype} (n={int(mask.sum())})")
        ax.legend(loc="upper right")
    for ax in axes[len(np.unique(labels)) :]:
        ax.axis("off")


def plot_weight_boxplot(
    adata: anndata.AnnData, cmp: int, ax: plt.Axes, celltype_col: str, role: str, color
) -> None:
    """Box plot (outliers hidden) of each cell type's `cmp` `role` weights."""
    df = pd.DataFrame(
        {
            "weight": adata.obsm[FACTOR_KEYS[role]][:, cmp - 1],
            "Cell Type": adata.obs[celltype_col].to_numpy(),
        }
    )
    sns.boxplot(
        data=df, x="weight", y="Cell Type", showfliers=False, color=color, ax=ax
    )
    limit = np.max(np.abs(ax.get_xticks()))
    ax.set(xticks=np.linspace(-limit, limit, num=5), xlabel="Cell Weight")
    ax.set_title(f"Cmp. {cmp} {role.capitalize()} Weight by Cell Type")


# =============================================================================
# LR pair interaction between sender and receiver cells
# =============================================================================


def plot_interaction_heatmap(
    adata: anndata.AnnData,
    cmp: int,
    ligand: str | list[str],
    receptor: str,
    ax: plt.Axes,
    sender_celltype: str | None,
    receiver_celltype: str | None,
    celltype_col: str,
) -> None:
    """Binned ligand x receptor expression product between sender cells
    (rows, highest `cmp` sender weight first) and receiver cells (columns,
    lowest receiver weight first), optionally restricted to one cell type
    each; a list `ligand` is a multi-subunit complex."""
    senders = adata
    receivers = adata
    if sender_celltype is not None:
        senders = adata[adata.obs[celltype_col] == sender_celltype]
    if receiver_celltype is not None:
        receivers = adata[adata.obs[celltype_col] == receiver_celltype]
    senders = senders[np.argsort(-senders.obsm["sc_B"][:, cmp - 1])]
    receivers = receivers[np.argsort(receivers.obsm["rc_C"][:, cmp - 1])]

    if isinstance(ligand, list):
        df = binned_complex_ligand_product_matrix(senders, receivers, ligand, receptor)
        ligand = "&".join(ligand)
    else:
        df = binned_expression_product_matrix(senders, receivers, ligand, receptor)

    sns.heatmap(df, ax=ax, cmap="viridis")
    ax.set_title(f"{ligand}-{receptor} Interaction")
    ax.set_xlabel(f"Receiver {receiver_celltype or 'Cells'}")
    ax.set_ylabel(f"Sender {sender_celltype or 'Cells'}")
    ax.set(xticks=[], yticks=[])


# =============================================================================
# Per-cell PaCMAP embeddings (the PaCMAP supplement's, for any component,
# eigenstate, or categorical label)
# =============================================================================


def pacmap_axes(ax: plt.Axes) -> None:
    """PaCMAP coordinates have no units: no ticks, equal aspect."""
    ax.set(xlabel="PaCMAP1", ylabel="PaCMAP2", xticks=[], yticks=[], aspect="equal")


def plot_value_pacmap(
    adata: anndata.AnnData,
    values: np.ndarray,
    ax: plt.Axes,
    title: str,
    cbar_label: str,
    saturation_percentile: float,
) -> None:
    """Cells on the PaCMAP colored by `values`, saturating at
    `saturation_percentile` of |values|."""
    x, y = np.asarray(adata.obsm["PaCMAP"]).T
    points = common.plot_value_scatter(ax, x, y, values, saturation_percentile)
    pacmap_axes(ax)
    # An inset follows the equal-aspect box rather than the full axes slot
    plt.colorbar(points, cax=ax.inset_axes((1.03, 0.0, 0.04, 1.0)), label=cbar_label)
    ax.set_title(title)


def plot_categorical_scatter(
    ax: plt.Axes, points: np.ndarray, labels: np.ndarray, colors: dict[str, str]
) -> None:
    """Cells colored by `labels` ({label: color}), drawn in random order so no
    label hides the others, with a legend outside the axes."""
    order = np.random.default_rng(0).permutation(len(labels))
    ax.scatter(
        *points[order].T,
        c=[colors[label] for label in labels[order]],
        s=style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
    present = set(labels)
    ax.legend(
        handles=[Patch(color=c, label=k) for k, c in colors.items() if k in present],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )


def plot_celltype_points(
    ax: plt.Axes,
    adata: anndata.AnnData,
    dataset: str,
    points: np.ndarray,
    selected: str | None = None,
    selected_scale: float = 1.0,
) -> None:
    """Cells colored by cell type (the manuscript's `plot_celltype_scatter`),
    the `selected` cell type's points drawn on top `selected_scale` times
    their diameter, with the cell type legend outside the axes."""
    x, y = points.T
    common.plot_celltype_scatter(adata, dataset, x, y, ax, thresholds=False)
    if selected is not None:
        celltypes = adata.obs[DATASETS[dataset].celltype_column].to_numpy()
        enlarge_points(
            ax,
            points[celltypes == selected],
            style.CELL_TYPE_COLORS[dataset][selected],
            selected_scale,
        )
    ax.legend(
        handles=common.celltype_handles(dataset),
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )


def enlarge_points(ax: plt.Axes, points: np.ndarray, colors, scale: float) -> None:
    """Redraw `points` on top at `scale` times a cell's diameter."""
    ax.scatter(
        *points.T,
        c=colors,
        s=scale**2 * style.POINT_SIZE_CELL,
        linewidths=0,
        rasterized=True,
    )
