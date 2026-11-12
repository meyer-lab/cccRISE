"""What each method keeps and loses when its cells are described more finely
(`figureS_comparator_labels`): PB's and STAC's coarse LR columns joined to
their fine ones, and CCC-RISE's at a reduced RISE rank (``uv run
cccrise-reduced-rise``) joined to the reported fit's, each pair of columns at
LR cos^2 `FLOW_FROM` or more, on the reported CCC-RISE fit's LR pairs.

Two marks tie the panels to the missed supplements: a coarse comparator
column associated with outcome (its strongest association of
`_missed.ASSOCIATION_NAMES` at q < `common.SIG_ALPHA`) that keeps no fine
column is boxed in red and named in bold; and a CCC-RISE column that keeps no
column of the other fit and whose carrying pairs
(`covariation.carrying_pairs`) are one LR pair has that pair named under it."""

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ... import covariation
from .. import common, style
from ._common import METHOD_COLORS, load
from ._data import short
from ._missed import ASSOCIATION_NAMES
from ._programs import lr_matrix
from ._recovery import LINK, panel_title

#: The comparators drawn: PB and STAC only.
COMPARATORS = ("PB", "STAC")
#: Columns are joined at this cos^2 or more.
FLOW_FROM = 0.09
BLOCK_STEP = 1.75
NAME_SIZE = style.FONT_SIZE_DENSE - 2.8
#: `run_reduced_rise`'s table
REDUCED = "reduced_rise_lr_loadings"
TITLES = {"cccRISE": "CCC-RISE"}
#: A red box's margin past its name (left) and its dot (right), in x units
BOX_PAD = (0.04, 0.07)


@dataclass(frozen=True, eq=False)
class Block:
    """One method's two fits: LR columns (pairs x columns) of the left (coarse
    or reduced RISE rank) and right (fine or reported) fit, the names under
    each side, and each fit's full name."""

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
    reported ones."""
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
            (f"RISE {low}", f"RISE {high}"),
            (
                f"CCC-RISE RISE {low}, CPD {cpd}",
                f"CCC-RISE RISE {high}, CPD {reported}",
            ),
        ),
    )


def _positions(cos2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Left columns top to bottom in order, right ones ordered by their best
    left match."""
    values = cos2.to_numpy()
    left = np.linspace(1, 0, len(cos2)) if len(cos2) > 1 else np.array([0.5])
    right = np.empty(cos2.shape[1])
    right[np.lexsort((-values.max(axis=0), values.argmax(axis=0)))] = np.linspace(
        1, 0, cos2.shape[1]
    )
    return left, right


def _names(ax, x: float, ys, columns, method: str, side: int) -> list:
    """`method`'s columns as dots at `x`, named beside them; the names."""
    ax.scatter([x] * len(ys), ys, s=3, color=METHOD_COLORS[method], lw=0, zorder=3)
    return [
        ax.text(
            x + 0.07 * side,
            y,
            short(method, column),
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=NAME_SIZE,
        )
        for y, column in zip(ys, columns, strict=True)
    ]


def _block(ax, block: Block, cos2: pd.DataFrame, x: float) -> list:
    """`block`'s left columns at `x`, its right ones at `x + 1`, joined where
    cos^2 >= `FLOW_FROM`; the left columns' names."""
    left, right = _positions(cos2)
    for (i, j), value in np.ndenumerate(cos2.to_numpy()):
        if value >= FLOW_FROM:
            ax.plot(
                [x, x + 1],
                [left[i], right[j]],
                color=METHOD_COLORS[block.method],
                lw=0.25 + 1.4 * value,
                alpha=0.75,
                solid_capstyle="butt",
                zorder=2,
            )
    names = _names(ax, x, left, cos2.index, block.method, -1)
    _names(ax, x + 1, right, cos2.columns, block.method, 1)
    title = TITLES.get(block.method, block.method)
    ax.text(x + 0.5, 1.07, title, ha="center", fontsize=style.FONT_SIZE_DENSE - 0.5)
    for position, end in zip((x, x + 1), block.ends, strict=True):
        ax.text(position, -0.11, end, ha="center", fontsize=style.FONT_SIZE_DENSE - 1)
    return names


def associated(dataset: str, method: str) -> pd.DataFrame:
    """Each of `method`'s coarse columns' strongest association of
    `ASSOCIATION_NAMES` (lowest q), indexed by column."""
    t = load("associations", "coarse", dataset)
    t = t[(t["Method"] == method) & t["Association"].isin(ASSOCIATION_NAMES)]
    return t.sort_values("PAdj").drop_duplicates("Component").set_index("Component")


def _box(ax, name, x: float) -> None:
    """A red box around `name` and its dot at `x`; `name` in bold."""
    name.set_fontweight("bold")
    extent = name.get_window_extent(ax.figure.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(extent.get_points())
    left, right = x0 - BOX_PAD[0], max(x1, x) + BOX_PAD[1]
    pad = 0.015
    ax.add_patch(
        Rectangle(
            (left, y0 - pad),
            right - left,
            y1 - y0 + 2 * pad,
            fc="none",
            ec=LINK,
            lw=0.8,
            zorder=4,
            clip_on=False,
        )
    )


def _boxed(dataset: str, block: Block, cos2: pd.DataFrame) -> np.ndarray:
    """Whether each coarse column of a comparator is boxed: associated
    (`associated`, q < `common.SIG_ALPHA`) and keeping no fine column."""
    q = associated(dataset, block.method)["PAdj"].reindex(cos2.index)
    return ((cos2.max(axis=1) < FLOW_FROM) & (q < common.SIG_ALPHA)).to_numpy()


def _name_lone(ax, block: Block, cos2: pd.DataFrame, x: float) -> None:
    """Under each CCC-RISE dot keeping no column of the other fit whose
    carrying pairs are one LR pair, that pair."""
    left, right = _positions(cos2)
    sides = (
        (x, left, block.left, cos2.max(axis=1)),
        (x + 1, right, block.right, cos2.max(axis=0)),
    )
    for at, heights, columns, kept in sides:
        for c, y in zip(kept.index, heights, strict=True):
            pairs = columns.index[covariation.carrying_pairs(columns[c].to_numpy())]
            if kept[c] < FLOW_FROM and len(pairs) == 1:
                ax.text(
                    at,
                    y - 0.035,
                    pairs[0].replace("^", "–").replace("&", "+"),
                    ha="center",
                    va="top",
                    fontsize=NAME_SIZE,
                    color="0.25",
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


def coarse_to_fine(ax, dataset: str, title: str) -> None:
    """Each block's left LR columns joined to its right ones where cos^2 >=
    `FLOW_FROM`, line width growing with cos^2; the missed supplements' marks
    (module docstring)."""
    pairs = lr_matrix(dataset, "cccRISE").index
    tables = {b: b.cos2(pairs) for b in blocks(dataset)}
    # Final limits first: `_box` measures names in data units
    ax.set(xlim=(-0.38, BLOCK_STEP * len(tables) - 0.37), ylim=(-0.15, 1.15))
    common.source_data(
        f"lr_cos2_between_fits_{dataset}",
        pd.concat([_cos2_rows(b, t) for b, t in tables.items()], ignore_index=True),
        f"LR cos^2 on {title} between each comparator's coarse and fine LR columns, "
        "and between CCC-RISE's LR columns at a reduced RISE rank and the reported "
        f"fit's, on the reported fit's LR pairs; a line is drawn at {FLOW_FROM} or "
        "more.",
    )
    for i, (block, cos2) in enumerate(tables.items()):
        x = BLOCK_STEP * i
        names = _block(ax, block, cos2, x)
        if block.method == "cccRISE":
            _name_lone(ax, block, cos2, x)
            continue
        for name, boxed in zip(names, _boxed(dataset, block, cos2), strict=True):
            if boxed:
                _box(ax, name, x)
    ax.set_axis_off()
    panel_title(ax, title, pad=2)
    ax.text(
        0.5,
        -0.04,
        "LR cos$^2$ match",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=style.FONT_SIZE_LABEL,
    )
