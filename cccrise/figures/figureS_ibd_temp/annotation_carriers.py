"""IBD draft: which fine types carry each cccRISE component, and whether the
comparators recover a component only when those types have their own label
(`run_lr_concordance`, `run_granularity_sampling`).

A. cccRISE's per-fine-type activity (share of a type's cells active in the
   component) as sender and receiver, for the types active in any component;
   boxes mark each component's most active sender and receiver. C4-C6 are sent
   by one fine type and received by another (EE -> EC, EE -> Tuft); the rest
   are sent and received by the same type.
B. Over the sampled labellings (s0, s12 / s24 / s36 x repeats, fine), each
   component's LR containment in each comparator's span, split by whether its
   most active sender and receiver types have a label of their own."""

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .. import common, style
from ..figureS_lr_concordance._common import METHODS, SHORT
from ._sensitivity import CONTAINED, SHORT_TYPES, activity, carriers, labels, load
from .granularity_sampling import METHOD_COLORS

MIN_ACTIVE = 0.3
SIDES = ("sender", "receiver")


def _shown_types(act: pd.DataFrame) -> list[str]:
    """Fine types active in some component (>= `MIN_ACTIVE`), by lineage."""
    best = act.groupby("celltype")["active_frac"].max()
    lineage = activity_lineage(act)
    shown = best[best >= MIN_ACTIVE].index
    return sorted(shown, key=lambda t: (lineage[t], t))


def activity_lineage(act: pd.DataFrame) -> pd.Series:
    isolation = load("isolation").drop_duplicates("celltype")
    return isolation.set_index("celltype")["coarse"].reindex(act["celltype"].unique())


def _carrier_heatmaps(axes, act, keys, names):
    types = _shown_types(act)
    lineage = activity_lineage(act)
    for ax, side in zip(axes, SIDES, strict=True):
        grid = (
            act[act["side"] == side]
            .pivot(index="component", columns="celltype", values="active_frac")
            .reindex(columns=types)
        )
        image = ax.imshow(grid, cmap=style.CMAPS["weight"], vmin=0, vmax=1, aspect="auto")
        for row, k in enumerate(grid.index):
            col = types.index(keys.loc[k, side])
            ax.add_patch(
                Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, lw=0.8,
                          ec=style.COLORS["ink"])
            )  # fmt: skip
        ax.set_yticks(range(len(grid)), [names[k] for k in grid.index])
        ax.set_title(f"cccRISE {side}s", fontsize=style.FONT_SIZE_TITLE)
        ax.grid(False)
    axes[-1].set_xticks(
        range(len(types)),
        [f"{SHORT_TYPES.get(t, t)} ({lineage[t]})" for t in types],
        rotation=90, fontsize=style.FONT_SIZE_DENSE,
    )  # fmt: skip
    axes[0].set_xticks(range(len(types)), [])
    common.colorbar(axes[0], image, "share of cells active")


def _category(row, keys) -> str:
    sender, receiver = keys.loc[row["cccrise_component"], list(SIDES)]
    s_own, r_own = row[f"{sender}"] == 1, row[f"{receiver}"] == 1
    if sender == receiver:
        return "own label" if s_own else "pooled"
    return {(True, True): "both", (True, False): "S only", (False, True): "R only"}.get(
        (s_own, r_own), "none"
    )


def _isolated_table(comps, keys):
    isolation = load("isolation").pivot(
        index="labelling", columns="celltype", values="share_of_group"
    )
    table = comps.join(isolation, on="labelling")
    return table.assign(category=table.apply(_category, axis=1, keys=keys))


def _by_isolation(axes, table, keys, names, order):
    for ax, k in zip(axes, order, strict=True):
        part = table[table["cccrise_component"] == k]
        sender, receiver = keys.loc[k, list(SIDES)]
        groups = (
            ["pooled", "own label"]
            if sender == receiver
            else ["none", "S only", "R only", "both"]
        )
        for m, method in enumerate(METHODS):
            sub = part[part["method"] == method]
            x = np.array([groups.index(c) for c in sub["category"]], float)
            jitter = np.random.default_rng(m).uniform(-0.08, 0.08, len(x))
            ax.scatter(
                x + (m - 1) * 0.22 + jitter, sub["lr_containment"],
                s=style.POINT_SIZE_BACKGROUND, color=METHOD_COLORS[method], lw=0,
                alpha=0.8, label=SHORT[method],
            )  # fmt: skip
        counts = part.drop_duplicates("labelling")["category"].value_counts()
        ax.set_xticks(
            range(len(groups)), [f"{c}\n(n={counts.get(c, 0)})" for c in groups],
            fontsize=style.FONT_SIZE_DENSE,
        )  # fmt: skip
        ax.axhline(CONTAINED, color=style.COLORS["reference"], lw=style.LINE_WIDTH_THIN,
                   ls=style.LINE_STYLE_REFERENCE)  # fmt: skip
        pair = (
            SHORT_TYPES.get(sender, sender)
            if sender == receiver
            else f"{SHORT_TYPES.get(sender, sender)} → {SHORT_TYPES.get(receiver, receiver)}"
        )
        ax.set_title(f"{names[k]}\n{pair}", fontsize=style.FONT_SIZE_DENSE + 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)
    for ax in axes[:: len(axes) // 2]:
        ax.set_ylabel("LR containment")
    axes[0].legend(fontsize=style.FONT_SIZE_DENSE - 1, loc="center left")


def plot(fig):
    act, keys, names = activity(), carriers(), labels()
    comps = load("sampling_components")
    comps = comps[comps["labelling"] != "_coarse"]
    top, bottom = fig.subfigures(2, 1, height_ratios=(1.1, 1))
    _carrier_heatmaps(top.subplots(2, 1), act, keys, names)
    # distinct sender and receiver first
    order = sorted(keys.index, key=lambda k: (keys.loc[k, "sender"] == keys.loc[k, "receiver"], k))
    axes = bottom.subplots(2, len(order) // 2, sharey=True).ravel()
    _by_isolation(axes, _isolated_table(comps, keys), keys, names, order)


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 190 * style.MM))
