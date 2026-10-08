"""Which cccRISE components the comparators recover.

Per cccRISE component and comparator fit (own rank; coarse and fine
annotation), the fit's column that best matches the component's top-10 LR
pairs. A match is scored over the pairs both top-10s share: per pair, the
smaller of its weight in the component and in the column (squared loading
relative to the column's top pair, so a top pair weighs 1), summed
(`shared_min_weight`). A shared top pair alone scores 1, however spread
either column is; columns heavy on different shared pairs score near 0.
Colour: that score for the best column (ties: more shared pairs), on a scale
to `VMAX` (just above the largest score) so matches of different strength
stay apart. Number: how many pairs it shares. Rows: comparator fits;
columns: cccRISE components."""

import numpy as np
import pandas as pd

from .. import common, style
from ._common import ANNOTATIONS, METHODS, SHORT, load

K = 10
VMAX = 1.6
FITS = [(a, m) for a in ANNOTATIONS for m in METHODS]


def best(table: pd.DataFrame) -> pd.DataFrame:
    """Per (cccRISE component, annotation, method): the column with the
    largest per-pair min weight on the shared top-`K` pairs (`min_weight`),
    then the most shared pairs."""
    table = table[(table["k"] == K) & table["method"].isin(METHODS)]
    table = table.sort_values(["min_weight", "shared"], ascending=False, kind="stable")
    keys = ["cccrise_component", "annotation", "method"]
    return table.drop_duplicates(keys).set_index(keys).sort_index()


def _grid(rows, comps, name):
    """``(fits, components)`` values of `name`."""
    return np.array([[rows.loc[(c, *fit), name] for c in comps] for fit in FITS])


def _annotate(ax, shared, score):
    for (i, j), n in np.ndenumerate(shared):
        ax.text(
            j, i, f"{int(n)}", ha="center", va="center",
            color="white" if score[i, j] > 0.75 * VMAX else style.COLORS["ink"],
            fontsize=style.FONT_SIZE_DENSE + 1, fontweight="bold",
        )  # fmt: skip


def plot(fig):
    rows = best(load("overlap"))
    comps = sorted(rows.index.get_level_values(0).unique())
    score = _grid(rows, comps, "min_weight")
    ax = fig.subplots()
    image = ax.imshow(
        score, cmap=style.CMAPS["weight"], vmin=0, vmax=VMAX, aspect="auto"
    )
    _annotate(ax, _grid(rows, comps, "shared"), score)
    ax.set_yticks(range(len(FITS)), [f"{SHORT[m]} {a}" for a, m in FITS])
    ax.set_xticks(range(len(comps)), [f"C{c}" for c in comps])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    ax.axhline(len(METHODS) - 0.5, color="white", lw=3)
    ax.set_xlabel("cccRISE component")
    ax.set_title("Recovery of cccRISE components", fontsize=style.FONT_SIZE_TITLE)
    ax.grid(False)
    fig.colorbar(
        image,
        ax=ax,
        label="weight on shared pairs\n(per-pair min; top pair = 1)",
        ticks=[0, 0.5, 1, 1.5],
        location="bottom",
        shrink=0.8,
        pad=0.02,
    )


def makeFigure():
    return common.standalone(plot, (style.FIGURE_WIDTH_DOUBLE / 2, 85 * style.MM))
