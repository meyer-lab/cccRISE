"""How much of each comparator's LR columns the other comparators' spans hold
(own-rank fits; ``||P_X a||^2`` for a unit column `a`). One 2x2 per method
whose columns are measured: rows are the annotation, columns the other two
methods. Colour and number: the mean containment of the method's columns;
below it, how many of them are contained (>= 0.85 and above the
row-permutation null's 95th percentile)."""

import numpy as np

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT, load


def _block(ax, table, source):
    targets = [m for m in METHODS if m != source]
    sub = table[table["source"] == source]
    grid = sub.groupby(["annotation", "target"]).agg(
        mean=("containment", "mean"), hits=("contained", "sum"), n=("contained", "size")
    )
    means = np.array([[grid.loc[(a, t), "mean"] for t in targets] for a in ANNOTATIONS])
    image = ax.imshow(
        means, cmap=style.CMAPS["sequential"], vmin=0, vmax=1, aspect="equal"
    )
    for (i, j), v in np.ndenumerate(means):
        row = grid.loc[(ANNOTATIONS[i], targets[j])]
        ax.text(
            j, i, f"{v:.2f}\n{int(row['hits'])} of {int(row['n'])}", ha="center",
            va="center", fontsize=style.FONT_SIZE_DENSE,
            color="white" if v > 0.6 else style.COLORS["ink"],
        )  # fmt: skip
    ax.set_xticks(range(len(targets)), [f"in {SHORT[t]}" for t in targets])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ANNOTATIONS)), ANNOTATIONS)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(f"{SHORT[source]} columns", fontsize=style.FONT_SIZE_TITLE, pad=16)
    ax.grid(False)
    return image


def plot(fig):
    table = load("pair_containment")
    axes = fig.subplots(1, len(METHODS))
    for ax, source in zip(axes, METHODS, strict=True):
        image = _block(ax, table, source)
    fig.colorbar(
        image, ax=list(axes), label="mean LR containment of the columns",
        location="bottom", shrink=0.5, pad=0.04, ticks=[0, 0.5, 1],
    )  # fmt: skip


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 70 * style.MM))
