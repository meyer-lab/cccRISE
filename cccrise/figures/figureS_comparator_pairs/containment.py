"""How much of each comparator column lies in each other comparator's span
(LR mode, own-rank fits; ``||P_X a||^2`` for a unit column `a`): rows are
every fit's columns, grouped by method; columns are the other methods; the
coarse fits beside the fine fits."""

import numpy as np

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT, load
from ..figureS_lr_concordance.containment import blocked_heatmap


def _grid(table, annotation):
    """Rows (with a blank row between methods), their labels and the method
    of each row (None for blanks), and the ``(rows, METHODS)`` values (NaN on
    a method's own column and on blanks)."""
    sub = table[table["annotation"] == annotation]
    wide = sub.pivot_table(
        index=["source", "component"], columns="target", values="containment"
    ).reindex(columns=METHODS)
    values, labels, owners = [], [], []
    for i, method in enumerate(METHODS):
        if i:
            values.append(np.full(len(METHODS), np.nan))
            labels.append("")
            owners.append(None)
        block = wide.loc[method]
        values += list(block.to_numpy())
        labels += [str(c) for c in block.index]
        owners += [method] * len(block)
    return np.array(values), labels, owners


def _heatmap(ax, table, annotation, length):
    values, labels, owners = _grid(table, annotation)
    groups = [SHORT[m] if m else None for m in owners]
    image = blocked_heatmap(ax, values, labels, groups, length, group_x=-0.2)
    ax.set_xticks(range(len(METHODS)), [f"in {SHORT[m]}" for m in METHODS])
    ax.xaxis.tick_top()
    ax.set_title(annotation, fontsize=style.FONT_SIZE_DENSE + 1)
    return image


def plot(fig):
    table = load("pair_containment")
    length = max(len(_grid(table, a)[0]) for a in ANNOTATIONS)
    axes = fig.subplots(1, 3, width_ratios=(1, 0.35, 1))
    axes[1].set_axis_off()
    for ax, annotation in zip(axes[::2], ANNOTATIONS, strict=True):
        image = _heatmap(ax, table, annotation, length)
    fig.colorbar(
        image, ax=list(axes), label="LR containment in the other method's span",
        location="bottom", shrink=0.6, pad=0.02, ticks=[0, 0.5, 1],
    )  # fmt: skip


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 90 * style.MM))
