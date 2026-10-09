"""How much of each comparator column the other comparators' spans hold
(own-rank fits; ``||P_X a||^2`` for a unit column `a`). One block per method
whose columns are measured: its coarse columns, then its fine columns
(rows, numbered), against the other two methods (columns). A block's height
follows the square root of its row count, so 6 and 20 rows end up at
similar heights without the short blocks stretched."""

import numpy as np

from .. import common, style
from ..figureS_lr_concordance._common import ANNOTATIONS, METHODS, SHORT, load
from ..figureS_lr_concordance.containment import blocked_heatmap

GRID_ROWS = 40  # grid rows the tallest block spans


def _block(table, source):
    """The source's columns (coarse, a blank row, fine) against the other
    methods: values, tick labels, annotation of every row (None: blank)."""
    targets = [m for m in METHODS if m != source]
    values, labels, groups = [], [], []
    for i, annotation in enumerate(ANNOTATIONS):
        if i:
            values.append(np.full(len(targets), np.nan))
            labels.append("")
            groups.append(None)
        sub = table[(table["source"] == source) & (table["annotation"] == annotation)]
        wide = sub.pivot_table(
            index="component", columns="target", values="containment"
        )[targets]
        values += list(wide.to_numpy())
        labels += [str(c) for c in wide.index]
        groups += [annotation] * len(wide)
    return np.array(values), labels, groups, targets


def plot(fig):
    table = load("pair_containment")
    blocks = {m: _block(table, m) for m in METHODS}
    tallest = max(len(b[0]) for b in blocks.values())
    # a label gutter before every block
    grid = fig.add_gridspec(GRID_ROWS + 5, 6, width_ratios=(0.45, 1, 0.45, 1, 0.45, 1))
    for i, (source, (values, labels, groups, targets)) in enumerate(blocks.items()):
        rows = round(GRID_ROWS * np.sqrt(len(values) / tallest))
        ax = fig.add_subplot(grid[:rows, 2 * i + 1])
        image = blocked_heatmap(ax, values, labels, groups, len(values), group_x=-0.3)
        ax.set_xticks(range(len(targets)), [f"in {SHORT[t]}" for t in targets])
        ax.xaxis.tick_top()
        ax.set_title(f"{SHORT[source]} columns", fontsize=style.FONT_SIZE_TITLE)
    fig.colorbar(
        image, cax=fig.add_subplot(grid[-2:, 2:5]), orientation="horizontal",
        label="LR containment in the other method's span", ticks=[0, 0.5, 1],
    )  # fmt: skip


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE, 90 * style.MM))
