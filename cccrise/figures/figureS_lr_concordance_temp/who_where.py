"""What, who, where (draft panel C): for every CP comparator column (own
rank; c = coarse, f = fine types) and its best top-10 match among cccRISE
components, the |cosine| of their LR columns, of their sender and receiver
cell-type profiles (cccRISE: mean sample-scaled per-cell loading per type),
and of their condition columns. Rows are grouped by the column's class
(shared top-10 pairs must carry >= 50% of the comparator column's loading)
and labelled in the matched component's colour. STACCato is left out: its
Tucker core does not tie a column to one sender, receiver and sample
profile."""

import numpy as np
import seaborn as sns

from .. import common, style
from ..figureS_lr_concordance._common import (
    CLASSES,
    SHORT,
    column_order,
    load,
    top_pair,
)

AGREEMENTS = {
    "lr_abs_cos": "LR",
    "sender_cos": "sender",
    "receiver_cos": "receiver",
    "condition_cos": "samples",
}
GROUPS = dict(
    zip(
        CLASSES,
        ("recovered", "same LR,\nother who/where", "one sample", "broad"),
        strict=True,
    )
)


def rows():
    """CP columns with their match, ordered by class, then as `column_order`."""
    matches = load("matches")
    matches = matches[matches["sender_cos"].notna()]
    order = {k: i for i, k in enumerate(column_order(matches))}
    keys = ["annotation", "method", "comparator_component"]
    matches = matches.assign(
        group=matches["lr_class_mass"].map(CLASSES.index),
        order=[order[k] for k in matches[keys].itertuples(index=False, name=None)],
    )
    return matches.sort_values(["group", "order"]).reset_index(drop=True)


def _label(row, pairs):
    key = (row.annotation, row.method, row.comparator_component)
    pair = pairs.get(key, "").replace("^", "–")
    return (
        f"{SHORT[row.method]} {row.annotation[0]}{row.comparator_component}"
        f" · {pair} → C{row.cccrise_component}"
    )


def _cells(ax, grid):
    for (i, j), v in np.ndenumerate(grid):
        ax.text(
            j, i, f"{v:.2f}", ha="center", va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if v > 0.6 else style.COLORS["ink"],
        )  # fmt: skip


def _groups(ax, table):
    """A white rule between classes and each class named right of its rows."""
    x = len(AGREEMENTS) - 0.5
    for group, sub in table.groupby("group"):
        top, bottom = sub.index.min() - 0.5, sub.index.max() + 0.5
        if top > 0:
            ax.axhline(top, color="white", lw=2.5)
        ax.plot(
            [x + 0.12] * 2, [top + 0.15, bottom - 0.15], color=style.COLORS["ink"],
            lw=style.LINE_WIDTH_THIN, clip_on=False,
        )  # fmt: skip
        ax.text(
            x + 0.22, (top + bottom) / 2, GROUPS[CLASSES[group]], va="center",
            fontsize=style.FONT_SIZE_DENSE,
        )  # fmt: skip


def plot(fig):
    table = rows()
    pairs = top_pair(load("columns"))
    grid = table[list(AGREEMENTS)].to_numpy(float)
    palette = sns.color_palette("Dark2", 5)
    ax = fig.subplots()
    image = ax.imshow(
        grid, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
    )
    _cells(ax, grid)
    ax.set_yticks(
        range(len(table)),
        [_label(r, pairs) for r in table.itertuples()],
        fontsize=style.FONT_SIZE_DENSE - 0.5,
    )
    for tick, c in zip(ax.get_yticklabels(), table["cccrise_component"], strict=True):
        tick.set_color(palette[c - 1])
    ax.set_xticks(range(len(AGREEMENTS)), list(AGREEMENTS.values()))
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    _groups(ax, table)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.colorbar(
        image, ax=ax, label="|cosine|", shrink=0.5, location="bottom", pad=0.02
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE * 0.6, 85 * style.MM))
