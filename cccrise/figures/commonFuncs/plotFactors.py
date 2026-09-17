from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ...utils import (
    binned_complex_ligand_product_matrix,
    binned_expression_product_matrix,
    celltype_component_association,
)
from ..common import getSetup, subplotLabel

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def plot_condition_factors(
    data: anndata.AnnData,
    ax: Axes,
    cond: str = "Condition",
    cond_group_labels: pd.Series | None = None,
    color_key=None,
    group_cond=False,
    normalize=False,
):
    """Plots condition factors"""
    # Get sample names in the order of condition_unique_idxs
    idxs = np.argsort(np.unique(data.obs["condition_unique_idxs"]))
    yt = pd.Series(np.unique(data.obs[cond]))[idxs]

    X = np.array(data.uns["A"])

    if normalize is True:
        XX = X
        X -= np.median(XX, axis=0)
        X /= np.std(XX, axis=0)
        X /= np.max(np.abs(X))

    if cond_group_labels is not None:
        # Align cond_group_labels with the reordered yt
        cond_group_labels = cond_group_labels.loc[yt].reset_index(drop=True)
        if group_cond is True:
            ind = cond_group_labels.argsort()
            cond_group_labels = cond_group_labels.iloc[ind]
            X = X[ind]
            yt = yt.iloc[ind]

        ax.tick_params(axis="y", which="major", pad=20, length=0)
        if color_key is None:
            colors = sns.color_palette(
                "Set2", n_colors=pd.Series(cond_group_labels).nunique()
            ).as_hex()
        else:
            colors = color_key
        lut = {}
        legend_elements = []
        for index, group in enumerate(pd.unique(cond_group_labels)):
            lut[group] = colors[index]
            legend_elements.append(Patch(color=colors[index], label=group))

        group_values = list(cond_group_labels)
        row_colors = [lut.get(val, "#cccccc") for val in group_values]
        ROW_RECTANGLE_X_OFFSET = -0.02
        ROW_RECTANGLE_WIDTH = 0.02
        # Add colored rectangles for each row
        for iii, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    xy=(ROW_RECTANGLE_X_OFFSET, iii),
                    width=ROW_RECTANGLE_WIDTH,
                    height=1,
                    color=color,
                    lw=0,
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
            )

        # Create legend outside the plot area
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Position legend to the right of the plot
            frameon=False,  # Remove frame for a cleaner look
            fontsize=10,
            title="Condition",
            title_fontsize=12,
        )

    xticks = np.arange(1, X.shape[1] + 1)
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel="Component")


def plot_eigenstate_factors(data: anndata.AnnData, ax: Axes, factor_type: str):
    """Plots eigenstate factors"""
    cp_rank = data.uns["B"].shape[1]
    xticks = np.arange(1, cp_rank + 1)
    X = data.uns["B"] if factor_type == "B" else data.uns["C"]
    X = X / np.max(np.abs(np.array(X)))
    rise_rank = data.uns["B"].shape[0]
    yt = np.arange(1, rise_rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")


STAT_LABELS = {
    "mean": "Mean Cell Loading",
    "effect_size": "Cohen's $d$ vs. Other Cells",
    "enrichment": "log$_2$ Enrichment in Top {top_perc:g}% of Cells",
    "pct_top": "% of Cells in Top {top_perc:g}%",
}


def plot_celltype_association(
    data: anndata.AnnData,
    ax: Axes,
    factor_type: str = "sender",
    celltype_col: str = "celltype",
    stat: str = "mean",
    top_perc: float = 10,
    normalize: str | None = "global",
    scale: float | None = None,
    show_counts: bool = False,
) -> pd.DataFrame:
    """Heatmap of cell types against components, colored by each cell type's
    association with each component.

    Unlike the pseudobulk decomposition, whose sender/receiver factors are
    themselves cell type x component matrices (`plot_eigenstate_factors`), CCC-RISE
    resolves the cell dimensions into RISE eigenstates; the associations plotted
    here are aggregated from the per-cell loadings by
    `celltype_component_association`, which documents the available `stat` options.

    Parameters
    ----------
    normalize : {"global", "component", None}, default="global"
        'global' divides the whole matrix by its largest magnitude, keeping
        components comparable to each other on a fixed -1 to 1 color scale.
        'component' rescales each component separately, which shows the ranking of
        cell types within a component but hides differences in component magnitude.
        None plots the raw values on symmetric limits.
    scale : float, optional
        Divisor to use in place of the matrix's own maximum magnitude when
        `normalize="global"`. Pass the same value across panels (e.g. the joint
        maximum of the sender and receiver matrices) to make their colors
        directly comparable, since normalization is otherwise per call.
    show_counts : bool, default=False
        Append each cell type's cell count to its label.

    Returns
    -------
    pandas.DataFrame
        The (unnormalized) association values that were plotted.
    """
    df = celltype_component_association(
        data,
        factor_type=factor_type,
        celltype_col=celltype_col,
        stat=stat,
        top_perc=top_perc,
    )
    values = df.to_numpy(dtype=float)

    if normalize == "global":
        values = values / (float(np.nanmax(np.abs(values))) if scale is None else scale)
    elif normalize == "component":
        values = values / np.nanmax(np.abs(values), axis=0, keepdims=True)
    elif normalize is not None:
        raise ValueError(f"Unknown normalize: {normalize}")

    yt = df.index.to_numpy()
    if show_counts:
        counts = data.obs[celltype_col].astype(str).value_counts()
        yt = np.array([f"{name} (n={counts[name]})" for name in yt])

    cbar_label = STAT_LABELS[stat].format(top_perc=top_perc)
    if normalize is not None:
        cbar_label = f"{cbar_label} (Normalized)"

    # Normalized values live on a fixed -1 to 1 scale, so a panel scaled by a
    # shared `scale` keeps color limits comparable rather than restretching them.
    vmax = 1.0 if normalize is not None else float(np.nanmax(np.abs(values)))
    sns.heatmap(
        data=values,
        xticklabels=df.columns.to_numpy(),
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": cbar_label},
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel="Component", ylabel="")

    return df


def plot_lr_factors(data: anndata.AnnData, ax: Axes, trim=True, weight=0.08):
    """Plots lr factors"""
    # Read the LR factor and pair information from .uns
    X = np.array(data.uns["D"])
    yt = data.uns["lr_pairs"]
    rank = X.shape[1]

    if trim is True:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > weight
        X = X[kept_idxs]
        yt = yt[kept_idxs]

    ind = reorder_table(X)
    X = X[ind]
    X = X / np.max(np.abs(X))
    yt = yt[ind]
    xticks = np.arange(1, rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")
    # A low trim threshold keeps enough LR pairs that their row labels
    # collide; shrink them with row count rather than letting a lenient
    # threshold produce an unreadable heatmap.
    ax.tick_params(axis="y", labelsize=float(np.clip(200 / max(len(yt), 1), 4, 9)))


def _lr_highlight_mask(weights: np.ndarray, highlight: int | str) -> np.ndarray:
    """Boolean mask over rank-sorted `weights` selecting the pairs to mark."""
    mask = np.zeros(weights.size, dtype=bool)
    if isinstance(highlight, int):
        mask[: min(highlight, weights.size)] = True
        return mask

    q1, q3 = np.percentile(weights, [25, 75])
    if highlight == "tukey":
        threshold = q3 + 1.5 * (q3 - q1)
    elif highlight == "q3":
        threshold = q3
    else:
        raise ValueError(f"Unknown highlight rule: {highlight}")
    return weights > threshold


def plot_lr_factor_rank(
    X: anndata.AnnData,
    cmp: int,
    ax: Axes,
    highlight: int | str = 5,
    include_negative: bool = False,
    max_labels: int = 10,
    label_fontsize: float = 9,
) -> pd.DataFrame:
    """Plot every LR pair's weight for one component against its rank in that
    component, marking and labeling only the pairs that stand out.

    The LR factor is sparse, so the ranked curve has a steep head, a long flat
    middle that carries no signal, and (since the factor is signed) a short tail
    below zero. Drawing the line for all pairs but markers only for the extremes
    keeps the shape of the component visible while naming just the pairs worth
    interpreting -- a more compact read than the full factor heatmap
    (`plot_lr_factors`) or the fixed top-N barplots (`plot_lr_factors_partial`).

    Parameters
    ----------
    cmp : int
        Component to plot (1-indexed).
    highlight : int or {"tukey", "q3"}, default=5
        Which pairs get markers. An int takes that many top-ranked pairs. "tukey"
        takes those above Q3 + 1.5 * IQR and "q3" those above the 75th percentile;
        note that on a factor this sparse the quartiles sit near zero, so both
        rules select a large fraction of the pairs.
    include_negative : bool, default=False
        Also mark the negative tail, applying the mirrored rule (bottom N, or below
        the lower fence / Q1). Worth enabling when a component's most negative
        weights rival its positive ones in magnitude.
    max_labels : int, default=10
        Cap on labeled pairs per tail; the rest keep their markers. Labels are laid
        out in a callout column and nudged apart so they never overlap, however
        tightly the marked weights are packed.

    Returns
    -------
    pandas.DataFrame
        One row per LR pair, sorted by descending weight, with `lr_pair`, `weight`,
        `rank`, and `highlighted` columns.
    """
    weights = np.asarray(X.uns["D"], dtype=float)[:, cmp - 1]
    names = np.asarray(X.uns["lr_pairs"]).astype(str)

    order = np.argsort(-weights)
    ranked = pd.DataFrame(
        {
            "lr_pair": names[order],
            "weight": weights[order],
            "rank": np.arange(1, weights.size + 1),
        }
    )

    top_mask = _lr_highlight_mask(ranked["weight"].to_numpy(), highlight)
    bottom_mask = np.zeros_like(top_mask)
    if include_negative:
        # Mirror the rule by ranking the flipped weights, then flipping back
        bottom_mask = _lr_highlight_mask(-ranked["weight"].to_numpy()[::-1], highlight)[
            ::-1
        ]
    ranked["highlighted"] = top_mask | bottom_mask

    colors = sns.color_palette("colorblind")
    ax.axhline(0.0, color="0.75", lw=0.6, ls="--", zorder=0)
    ax.plot(ranked["rank"], ranked["weight"], color="0.35", lw=1.0, zorder=1)

    # Fix the limits from the data before annotating: the callouts are laid out in
    # data coordinates and setting limits at all disables autoscaling, so they have
    # to be established up front rather than left to a later autoscale.
    lo, hi = float(weights.min()), float(weights.max())
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, len(ranked) + 1)

    for mask, color, from_top in [
        (top_mask, colors[3], True),
        (bottom_mask, colors[0], False),
    ]:
        selected = ranked[mask]
        if selected.empty:
            continue
        ax.scatter(
            selected["rank"],
            selected["weight"],
            s=20,
            color=color,
            zorder=2,
            clip_on=False,
        )
        labeled = selected.head(max_labels) if from_top else selected.tail(max_labels)
        _annotate_lr_callouts(ax, selected, labeled, color, from_top, label_fontsize)

    ax.set(
        xlabel="LR Pair Rank",
        ylabel=f"Cmp. {cmp} Weight",
        title=f"Cmp. {cmp} Ligand-Receptor Pair Weights by Rank",
    )

    return ranked


def _annotate_lr_callouts(
    ax: Axes,
    selected: pd.DataFrame,
    labeled: pd.DataFrame,
    color,
    from_top: bool,
    fontsize: float,
) -> None:
    """Label marked points in a callout column, nudged apart to avoid collisions.

    Points at the head of a sparse factor are packed into a tiny weight range, so
    labels placed at their own heights would overlap; each is instead pushed at
    least one label-height away from the previous one, and the axis is extended if
    the resulting column runs past it. The column is anchored just beyond the last
    marked point so it clears the markers even when the rule marks many pairs.
    """
    x0, x1 = ax.get_xlim()
    lo, hi = ax.get_ylim()
    span = hi - lo
    gap = 0.07 * span

    names = labeled["lr_pair"].to_numpy().astype(str)
    label_weights = labeled["weight"].to_numpy(dtype=float)
    ranks = labeled["rank"].to_numpy(dtype=float)
    if not from_top:
        # Walk the negative tail from the most negative label upward
        names, label_weights, ranks = names[::-1], label_weights[::-1], ranks[::-1]

    positions = []
    previous = None
    for weight in label_weights:
        if previous is None:
            position = float(weight)
        elif from_top:
            position = min(float(weight), previous - gap)
        else:
            position = max(float(weight), previous + gap)
        positions.append(position)
        previous = position

    # Extend the axis if the stacked column overruns it
    if from_top and positions[-1] < lo:
        ax.set_ylim(bottom=positions[-1] - 0.04 * span)
    elif not from_top and positions[-1] > hi:
        ax.set_ylim(top=positions[-1] + 0.04 * span)

    offset = 0.03 * (x1 - x0)
    label_x = (
        float(selected["rank"].max()) + offset
        if from_top
        else float(selected["rank"].min()) - offset
    )
    for name, weight, rank, position in zip(
        names, label_weights, ranks, positions, strict=True
    ):
        ax.annotate(
            f"{name} ({weight:+.3f})",
            xy=(rank, weight),
            xytext=(label_x, position),
            textcoords="data",
            fontsize=fontsize,
            va="center",
            ha="left" if from_top else "right",
            color=color,
            # Backing box so the flat middle of the curve does not strike through a
            # callout that lands near zero
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.4,
                "color": "0.55",
                "shrinkA": 1,
                "shrinkB": 2,
            },
        )


def plot_lr_factors_partial(
    X: anndata.AnnData, cmp: int, ax: Axes, geneAmount: int = 5, top=True
):
    """Plotting weights for lr factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"
    lr_factor = np.array(X.uns["D"])
    yt = X.uns["lr_pairs"]
    df = pd.DataFrame(data=lr_factor[:, cmp - 1], index=yt, columns=pd.Index([cmpName]))
    df = df.reset_index(names="LR")
    df = df.sort_values(by=cmpName)

    if top:
        sns.barplot(data=df.iloc[-geneAmount:, :], x="LR", y=cmpName, color="k", ax=ax)
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="LR", y=cmpName, color="k", ax=ax)

    ax.tick_params(axis="x", rotation=90)


def make_lr_factors_partial_figure(
    data_or_path: str | Path | anndata.AnnData,
    grid: tuple[int, int],
    figsize: tuple[int, int] = (12, 12),
) -> Figure:
    """Build a grid figure of top/bottom LR factor weights for every component."""
    from ...factors import load_factors_to_anndata

    ax, f = getSetup(figsize, grid)
    subplotLabel(ax)

    if isinstance(data_or_path, anndata.AnnData):
        X = data_or_path
    elif str(data_or_path).endswith(".nc"):
        X = load_factors_to_anndata(data_or_path)
    else:
        X = anndata.read_h5ad(data_or_path)

    for i in range(X.uns["A"].shape[1]):
        plot_lr_factors_partial(X, i + 1, ax[2 * i], geneAmount=10, top=True)
        plot_lr_factors_partial(X, i + 1, ax[2 * i + 1], geneAmount=10, top=False)

    return f


def reorder_table(projs: np.ndarray):
    """Reorder a table's rows using hierarchical clustering"""
    # Clean non-finite values
    clean_projs = np.nan_to_num(projs, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure no zero vectors (which cause issues with cosine distance)
    zero_rows = np.all(clean_projs == 0, axis=1)
    if np.any(zero_rows):
        # Add a small epsilon to zero rows to avoid cosine distance issues
        clean_projs[zero_rows, :] = 1e-10

    Z = sch.linkage(
        clean_projs, method="complete", metric="cosine", optimal_ordering=True
    )
    return sch.leaves_list(Z)


def plot_pair_lr_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(([X.uns["D"][:, cmp1 - 1]], [X.uns["D"][:, cmp2 - 1]]))
    df = pd.DataFrame(
        data=cmpWeights.transpose(),
        columns=pd.Index([f"Cmp. {cmp1}", f"Cmp. {cmp2}"]),
    )
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax, color="k")
    ax.set(title="LR Factors")


def plot_interaction_heatmap(
    X: anndata.AnnData,
    cmp: int,
    ligand: str | list[str],
    receptor: str,
    ax: Axes,
    sender_celltype: str | None = None,
    receiver_celltype: str | None = None,
    celltype_col: str = "celltype",
    cmap: str = "viridis",
    vmax: float | None = None,
    title: str | None = None,
    sender_label: str | None = None,
    receiver_label: str | None = None,
):
    """Plot binned expression product heatmap between sender and receiver cells for a component."""
    if sender_celltype is not None:
        X_sender = X[X.obs[celltype_col] == sender_celltype]
    else:
        X_sender = X

    if receiver_celltype is not None:
        X_receiver = X[X.obs[celltype_col] == receiver_celltype]
    else:
        X_receiver = X

    # Sort sender cells high to low by sc_B
    X_sender = X_sender[np.argsort(-X_sender.obsm["sc_B"][:, cmp - 1])]
    # Sort receiver cells low to high by rc_C
    X_receiver = X_receiver[np.argsort(X_receiver.obsm["rc_C"][:, cmp - 1])]

    if isinstance(ligand, list):
        df = binned_complex_ligand_product_matrix(
            X_sender, X_receiver, ligand, receptor
        )
        lig_name = "&".join(ligand)
    else:
        df = binned_expression_product_matrix(X_sender, X_receiver, ligand, receptor)
        lig_name = ligand

    sns.heatmap(df, ax=ax, cmap=cmap, vmax=vmax)

    if title is None:
        ax.set_title(f"{lig_name}-{receptor} Interaction")
    else:
        ax.set_title(title)

    if receiver_label is not None:
        ax.set_xlabel(receiver_label)
    elif receiver_celltype is not None:
        ax.set_xlabel(f"Receiver {receiver_celltype}")
    else:
        ax.set_xlabel("Receiver Cells")

    if sender_label is not None:
        ax.set_ylabel(sender_label)
    elif sender_celltype is not None:
        ax.set_ylabel(f"Sender {sender_celltype}")
    else:
        ax.set_ylabel("Sender Cells")

    ax.set_xticks([])
    ax.set_yticks([])
