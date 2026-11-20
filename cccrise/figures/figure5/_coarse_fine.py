"""What each method keeps and loses when its cells are described more finely
(Figure 5E-F): PB's and STAC's coarse LR columns joined to their fine ones,
and CCC-RISE's at a reduced RISE rank (``uv run cccrise-reduced-rise``) joined
to the reported fit's, each pair of columns at LR cos^2 `FLOW_FROM` or more,
on the reported CCC-RISE fit's LR pairs.

Each method is two rows, its left fit's columns (coarse, or the reduced RISE
rank) over its right fit's (fine, or the reported rank), PB, STAC and
CCC-RISE stacked top to bottom so that the cohorts' panels line up row for
row. Each panel names its rows (C / F, or CCC-RISE's RISE ranks, which differ
by cohort); the first also names the methods. The cohort is a bar in its
color under the panel, as in Figure 5G. A CCC-RISE column that keeps no
column of the other fit and whose carrying pairs (`covariation.carrying_pairs`)
are one LR pair has that pair named beside it."""

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from ... import covariation
from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS, METHOD_COLORS, load
from ._data import short
from ._programs import lr_matrix

#: The comparators drawn: PB and STAC only.
COMPARATORS = ("PB", "STAC")
#: Columns are joined at this cos^2 or more.
FLOW_FROM = 0.09
#: `run_reduced_rise`'s table
REDUCED = "reduced_rise_lr_loadings"
TITLES = {"cccRISE": "CCC-RISE"}
#: A method's two rows apart, and one method's top row to the next's (y units)
ROW_GAP, BLOCK_STEP = 0.8, 1.55
#: Names' clearance from their dot (y units)
NAME_PAD = 0.1
#: The cohort bar under the panel, below the last row's names (y units)
BAR_DROP = 0.55
BORDER_DROP = 0.3  # the border's bottom, below the last row
TOP = 0.3
X_PAD = 0.05  # beyond the outermost dots, inside the border
NAME_SIZE = 4.2  # pt: as large as the most crowded row allows
LONE_SIZE = style.FONT_SIZE_DENSE - 1.5
END_SIZE = style.FONT_SIZE_DENSE - 1
#: A lone pair's name: offset from its dot (x, y units), on a leader line
LONE_OFFSET = (0.14, -0.5)
#: Right-fit columns swapped by hand, per (cohort, method): IBD RISE 60's C4
#: and C5, which clears room under RISE 30's C1 for its pair's name.
SWAPS = {("ibd", "cccRISE"): (4, 5)}


@dataclass(frozen=True, eq=False)
class Block:
    """One method's two fits: LR columns (pairs x columns) of the left (coarse
    or reduced RISE rank) and right (fine or reported) fit, each row's label,
    and each fit's full name."""

    method: str
    left: pd.DataFrame
    right: pd.DataFrame
    ends: tuple[str, str]
    fits: tuple[str, str]

    def cos2(self, pairs: pd.Index) -> pd.DataFrame:
        """Left x right LR cos^2 on `pairs`."""
        left, right = (
            _unit(m.reindex(pairs).fillna(0)) for m in (self.left, self.right)
        )
        return pd.DataFrame(
            (left.to_numpy().T @ right.to_numpy()) ** 2,
            index=left.columns.astype(int),
            columns=right.columns.astype(int),
        )


def _unit(columns: pd.DataFrame) -> pd.DataFrame:
    return columns / np.linalg.norm(columns.to_numpy(), axis=0, keepdims=True)


@cache
def blocks(dataset: str) -> tuple[Block, ...]:
    """PB's and STAC's coarse and fine fits, then CCC-RISE's reduced and
    reported ones (rows labeled by RISE rank)."""
    reduced = load(REDUCED, dataset=dataset)
    ((low, cpd),) = (
        reduced[["RiseRank", "CpdRank"]].drop_duplicates().itertuples(index=False)
    )
    high, reported = common.selected_ranks(dataset)
    return (
        *(
            Block(
                m,
                lr_matrix(dataset, m, "coarse"),
                lr_matrix(dataset, m, "fine"),
                ("C", "F"),
                (f"{m} coarse", f"{m} fine"),
            )
            for m in COMPARATORS
        ),
        Block(
            "cccRISE",
            reduced.pivot(index="LRPair", columns="Component", values="Loading"),
            lr_matrix(dataset, "cccRISE"),
            (str(low), str(high)),
            (
                f"CCC-RISE RISE {low}, CPD {cpd}",
                f"CCC-RISE RISE {high}, CPD {reported}",
            ),
        ),
    )


def _positions(cos2: pd.DataFrame, swap=None) -> tuple[np.ndarray, np.ndarray]:
    """Left columns left to right in order, right ones under their best left
    match, `swap`'s two right columns exchanged."""
    values = cos2.to_numpy()
    left = np.linspace(0, 1, len(cos2)) if len(cos2) > 1 else np.array([0.5])
    right = np.empty(cos2.shape[1])
    right[np.lexsort((-values.max(axis=0), values.argmax(axis=0)))] = np.linspace(
        0, 1, cos2.shape[1]
    )
    if swap:
        i, j = (list(cos2.columns).index(c) for c in swap)
        right[[i, j]] = right[[j, i]]
    return left, right


def _row(ax, xs, y: float, columns, method: str, above: bool) -> None:
    """`method`'s columns as dots on row `y`, named above or below them."""
    ax.scatter(xs, [y] * len(xs), s=3, color=METHOD_COLORS[method], lw=0, zorder=3)
    for x, column in zip(xs, columns, strict=True):
        ax.text(
            x,
            y + (NAME_PAD if above else -NAME_PAD),
            short(method, column),
            ha="center",
            va="bottom" if above else "top",
            fontsize=NAME_SIZE,
        )


def _lone(ax, block: Block, cos2: pd.DataFrame, xs, ys) -> None:
    """Each CCC-RISE dot keeping no column of the other fit whose carrying
    pairs are one LR pair: that pair named between the rows, on a leader
    line to it."""
    sides = (
        (xs[0], ys[0], block.left, cos2.max(axis=1), 1),
        (xs[1], ys[1], block.right, cos2.max(axis=0), -1),
    )
    dx, dy = LONE_OFFSET
    for at, y, columns, kept, down in sides:
        for c, x in zip(kept.index, at, strict=True):
            pairs = columns.index[covariation.carrying_pairs(columns[c].to_numpy())]
            if kept[c] < FLOW_FROM and len(pairs) == 1:
                ax.annotate(
                    pairs[0].replace("^", "–").replace("&", "+"),
                    (x, y),
                    xytext=(min(x + dx, 1 - dx), y + down * dy),
                    ha="center",
                    va="center",
                    fontsize=LONE_SIZE,
                    color="0.25",
                    arrowprops={
                        "arrowstyle": "-",
                        "lw": 0.4,
                        "color": "0.4",
                        "shrinkA": 1,
                        "shrinkB": 1.5,
                    },
                )


def _row_labels(ax, rows: dict, method_names: bool) -> None:
    """Each row's label as a y tick (`rows`: {block: its two rows' heights}),
    and, if `method_names`, each method left of them."""
    ax.set_xticks([])
    ax.set_yticks(
        [y for ys in rows.values() for y in ys],
        [end for block in rows for end in block.ends],
        fontsize=END_SIZE,
    )
    ax.spines[:].set_visible(False)  # `common.data_border` boxes the rows instead
    if not method_names:
        return
    renderer = ax.figure.canvas.get_renderer()
    width = max(t.get_window_extent(renderer).width for t in ax.get_yticklabels())
    offset = style.TICK_LENGTH + 1.5 + width * 72 / ax.figure.dpi + 4  # pt
    for block, ys in rows.items():
        ax.annotate(
            TITLES.get(block.method, block.method),
            (0, np.mean(ys)),
            xycoords=("axes fraction", "data"),
            xytext=(-offset, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold",
        )


def _block(ax, dataset: str, block: Block, cos2, y: float) -> tuple[float, float]:
    """`block`'s left fit on row `y`, its right fit `ROW_GAP` below, joined
    where cos^2 >= `FLOW_FROM`, line width growing with cos^2; returns the
    rows' heights."""
    (left, right), ys = (
        _positions(cos2, SWAPS.get((dataset, block.method))),
        (y, y - ROW_GAP),
    )
    for (i, j), value in np.ndenumerate(cos2.to_numpy(dtype=float)):
        value = float(value)
        if value >= FLOW_FROM:
            ax.plot(
                [left[i], right[j]],
                ys,
                color=METHOD_COLORS[block.method],
                lw=0.25 + 1.4 * value,
                alpha=0.75,
                solid_capstyle="butt",
                zorder=2,
            )
    _row(ax, left, ys[0], cos2.index, block.method, above=True)
    _row(ax, right, ys[1], cos2.columns, block.method, above=False)
    if block.method == "cccRISE":
        _lone(ax, block, cos2, (left, right), ys)
    return ys


def _cohort_bar(ax, dataset: str, y: float) -> None:
    """The cohort's bar at `y` across the panel, its name under it."""
    ax.plot(
        [-X_PAD, 1 + X_PAD],
        [y, y],
        color=DATASET_COLORS[dataset],
        lw=2.5,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.annotate(
        DATASET_LABELS[dataset],
        (0.5, y),
        xytext=(0, -3),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=style.FONT_SIZE_DENSE,
    )


def _cos2_rows(block: Block, cos2: pd.DataFrame) -> pd.DataFrame:
    """`cos2` long, the columns named as drawn."""
    t = cos2.stack().rename("LR cos2").rename_axis(["From", "To"]).reset_index()
    return pd.DataFrame(
        {
            "From fit": block.fits[0],
            "From column": [short(block.method, c) for c in t["From"]],
            "To fit": block.fits[1],
            "To column": [short(block.method, c) for c in t["To"]],
            "LR cos2": t["LR cos2"],
        }
    )


def coarse_to_fine(ax, dataset: str, method_names: bool) -> None:
    """Each method's left fit's LR columns over its right fit's (module
    docstring); `method_names` names the methods left of the rows."""
    pairs = lr_matrix(dataset, "cccRISE").index
    tables = {b: b.cos2(pairs) for b in blocks(dataset)}
    bottom = -BLOCK_STEP * (len(tables) - 1) - ROW_GAP
    ax.set(xlim=(-X_PAD, 1 + X_PAD), ylim=(bottom - BAR_DROP, TOP))
    common.source_data(
        f"lr_cos2_between_fits_{dataset}",
        pd.concat([_cos2_rows(b, t) for b, t in tables.items()], ignore_index=True),
        f"LR cos^2 on {DATASET_LABELS[dataset]} between each comparator's coarse and "
        "fine LR columns, and between CCC-RISE's LR columns at a reduced RISE rank and "
        "the reported fit's, on the reported fit's LR pairs; a line is drawn at "
        f"{FLOW_FROM} or more.",
    )
    rows = {
        block: _block(ax, dataset, block, cos2, -BLOCK_STEP * i)
        for i, (block, cos2) in enumerate(tables.items())
    }
    _row_labels(ax, rows, method_names)
    common.data_border(ax, bottom - BORDER_DROP, TOP)
    _cohort_bar(ax, dataset, bottom - BAR_DROP)
