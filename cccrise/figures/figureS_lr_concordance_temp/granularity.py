"""Drawing for the subtype granularity draft (`granularity_subtypes`):
where a comparator column and its matched cccRISE component agree on LR
pairs, which cell subtypes carry the program (`run_lr_concordance` cell
tables).

One row per match, sender and receiver side by side. The x axis is the 26
fine cell types grouped (bands) under their coarse type. Bars: cccRISE's
mean sample-scaled loading per fine type / the largest type's. Black step:
the comparator column's |weight| / its largest, one step per coarse type for
a coarse fit (one weight for every fine type under it), per fine type for a
fine fit."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import style
from ..figureS_lr_concordance._common import SHORT, load

#: (annotation, method, comparator column, cccRISE component): matches whose
#: LR pairs agree (lr_class_min recovered or same LR), one per component.
MATCHES = (
    ("coarse", "Pseudobulk CPD", 2, 1),
    ("fine", "Tensor-cell2cell", 5, 2),
    ("coarse", "Pseudobulk CPD", 3, 3),
)
GAP = 0.8  # bar widths between coarse groups
SIDES = ("sender", "receiver")


def layout() -> pd.DataFrame:
    """Fine types grouped by coarse type, with their x positions."""
    types = load("celltypes").sort_values(["broad_cell_type", "celltype"])
    group = types["broad_cell_type"].factorize()[0]
    return types.assign(group=group, x=np.arange(len(types)) + GAP * group)


def _comparator(weights, types, annotation, method, column, side):
    """The comparator column's |weight| / max on every fine type."""
    sub = (
        weights[
            (weights["annotation"] == annotation)
            & (weights["method"] == method)
            & (weights["component"] == column)
            & (weights["side"] == side)
        ]
        .set_index("celltype")["weight"]
        .abs()
    )
    key = "broad_cell_type" if annotation == "coarse" else "celltype"
    return (sub / sub.max()).reindex(types[key]).to_numpy()


def _bands(ax, types):
    for g, sub in types.groupby("group"):
        if g % 2:
            ax.axvspan(
                sub["x"].min() - 0.5, sub["x"].max() + 0.5, color="0.95", zorder=0
            )


def _panel(ax, types, bars, step, color):
    _bands(ax, types)
    ax.bar(types["x"], bars, width=0.8, color=color, zorder=1)
    for x, y in zip(types["x"], step, strict=True):
        if y > 0.01:
            ax.plot([x - 0.5, x + 0.5], [y, y], color="black", lw=1.2, zorder=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(types["x"].min() - 0.7, types["x"].max() + 0.7)
    ax.grid(False)


def plot(fig, measure):
    """`measure(component, side, types)`: the cccRISE bar heights."""
    types = layout()
    weights = load("cell_weights")
    palette = sns.color_palette("Dark2", 5)
    axes = fig.subplots(len(MATCHES), len(SIDES), sharex=True, sharey=True)
    for row, (annotation, method, column, component) in zip(axes, MATCHES, strict=True):
        for ax, side in zip(row, SIDES, strict=True):
            step = _comparator(weights, types, annotation, method, column, side)
            _panel(
                ax, types, measure(component, side, types), step, palette[component - 1]
            )
            ax.set_title(
                f"C{component} vs {SHORT[method]} {annotation} {column}: {side}",
                fontsize=style.FONT_SIZE_DENSE + 0.5,
            )
    fig.legend(
        handles=[
            Patch(color="0.6", label="cccRISE (component colour)"),
            Line2D([], [], color="black", lw=1.2, label="comparator |weight| / max"),
        ],
        loc="outside lower center", ncol=2, frameon=False,
        fontsize=style.FONT_SIZE_DENSE,
    )  # fmt: skip
    for ax in axes[-1]:
        ax.set_xticks(
            types["x"], types["celltype"], rotation=90,
            fontsize=style.FONT_SIZE_DENSE - 1.5,
        )  # fmt: skip
    return axes


def mean_loading(component, side, types):
    """cccRISE's mean sample-scaled loading per fine type / the largest."""
    w = load("cell_weights")
    sub = w[
        (w["method"] == "cccRISE")
        & (w["annotation"] == "fine")
        & (w["component"] == component)
        & (w["side"] == side)
    ].set_index("celltype")["weight"]
    return np.clip(sub / sub.max(), 0, None).reindex(types["celltype"]).to_numpy()
