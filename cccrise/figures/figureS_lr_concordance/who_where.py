"""What, who, where: for every CP comparator column (own rank; c = coarse,
f = fine types) and its best top-10 match among cccRISE components, the
|cosine| of their LR columns, of their sender and receiver cell-type
profiles (cccRISE: mean sample-scaled per-cell loading per type), and of
their condition columns; right, the column's class (shared top-10 pairs
must carry >= 50% of the comparator column's loading)."""

import numpy as np

from .. import common, style
from ._common import CLASSES, column_label, column_order, load, top_pair

AGREEMENTS = {
    "lr_abs_cos": "LR",
    "sender_cos": "sender",
    "receiver_cos": "receiver",
    "condition_cos": "samples",
}
SHORT_CLASS = dict(
    zip(
        CLASSES,
        ("recovered", "same LR, other who/where", "one sample", "broad"),
        strict=True,
    )
)


def plot(fig):
    matches = load("matches")
    matches = matches[matches["sender_cos"].notna()]
    keys = column_order(matches)
    pairs = top_pair(load("columns"))
    rows = matches.set_index(["annotation", "method", "comparator_component"]).loc[keys]
    grid = rows[list(AGREEMENTS)].to_numpy(float)
    ax = fig.subplots()
    image = ax.imshow(
        grid, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="auto"
    )
    for (i, j), v in np.ndenumerate(grid):
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
            color="white" if v > 0.6 else style.COLORS["ink"],
        )
    labels = [
        f"{column_label(k, pairs)} → C{int(r.cccrise_component)}"
        for k, r in zip(keys, rows.itertuples(), strict=True)
    ]
    ax.set_yticks(range(len(keys)), labels, fontsize=style.FONT_SIZE_DENSE - 1)
    ax.set_xticks(
        range(len(AGREEMENTS)), list(AGREEMENTS.values()), rotation=40, ha="left"
    )
    ax.xaxis.tick_top()
    for i, cls in enumerate(rows["lr_class_mass"]):
        ax.text(
            len(AGREEMENTS) - 0.4,
            i,
            SHORT_CLASS[cls],
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1,
        )
    ax.grid(False)
    fig.colorbar(
        image, ax=ax, label="|cosine|", shrink=0.5, location="bottom", pad=0.02
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
