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
color over the panel, as in Figure 5G (`cohort_bar`). Only the columns with an outcome
association at q < `common.SIG_ALPHA` are named, in bold, and drawn larger:
PB's and STAC's columns and the reported CCC-RISE fit's components
(`_missed.associated`), and the reduced fit's components (`reduced_associated`,
tested as every fit is by `run_comparator_scores._associations`)."""

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from .. import common, style
from ._common import DATASET_LABELS, METHOD_COLORS, cohort_bar, load
from ._data import short
from ._missed import ASSOCIATION_NAMES, associated
from ._programs import lr_matrix

#: The comparators drawn: PB and STAC only.
COMPARATORS = ("PB", "STAC")
#: Columns are joined at this cos^2 or more.
FLOW_FROM = 0.09
#: `run_reduced_rise`'s table
REDUCED = "reduced_rise_lr_loadings"
TITLES = {"cccRISE": "CCC-RISE"}
#: A comparator block's left and right fit
LABELS = ("coarse", "fine")
#: A method's two rows apart, and one method's top row to the next's (y units)
ROW_GAP, BLOCK_STEP = 0.8, 1.55
#: Names' clearance from their dot (y units)
NAME_PAD = 0.1
BORDER_DROP = 0.5  # the border's bottom, below the last row
TOP = 0.95  # the border's top, above the first row: room for the key
X_PAD = 0.09  # beyond the outermost dots, inside the border
NAME_SIZE = style.FONT_SIZE_DENSE - 1  # pt: as Figure 5J's names
END_SIZE = style.FONT_SIZE_DENSE - 1
#: Dot area (pt^2): every column, and those named
DOT, NAMED_DOT = 3, 10
#: The size key's dots and text (pt), as Figure 5G's key, and its left edge
#: (axes fraction): in from the left, not against it
KEY_COLOR, KEY_SIZE = "0.4", style.FONT_SIZE_DENSE - 1
KEY_X = 0.06
#: `run_reduced_rise`'s associations table
REDUCED_ASSOCIATIONS = "reduced_rise_associations"


def reduced_associated(dataset: str) -> pd.DataFrame:
    """The reduced CCC-RISE fit's associations at q < `common.SIG_ALPHA`
    (``uv run cccrise-reduced-rise``)."""
    t = load(REDUCED_ASSOCIATIONS, dataset=dataset)
    return t[t["Association"].isin(ASSOCIATION_NAMES) & (t["PAdj"] < common.SIG_ALPHA)]


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

    def associated(self, dataset: str) -> tuple[set[int], set[int]]:
        """The left and right fit's columns with an association at q <
        `common.SIG_ALPHA`."""
        t = associated(dataset)
        t = t[t["Method"] == self.method]
        if self.method == "cccRISE":
            return set(reduced_associated(dataset)["Component"]), set(t["Component"])
        coarse, fine = (
            set(t.loc[t["Labelling"] == lab, "Component"]) for lab in LABELS
        )
        return coarse, fine

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


def _positions(cos2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Left columns left to right in order, right ones under their best left
    match."""
    values = cos2.to_numpy()
    left = np.linspace(0, 1, len(cos2)) if len(cos2) > 1 else np.array([0.5])
    right = np.empty(cos2.shape[1])
    right[np.lexsort((-values.max(axis=0), values.argmax(axis=0)))] = np.linspace(
        0, 1, cos2.shape[1]
    )
    return left, right


def _row(ax, xs, y: float, columns, method: str, above: bool, named) -> None:
    """`method`'s columns as dots on row `y`, those in `named` larger and
    named in bold above or below them."""
    sizes = [NAMED_DOT if c in named else DOT for c in columns]
    ax.scatter(xs, [y] * len(xs), s=sizes, color=METHOD_COLORS[method], lw=0, zorder=3)
    for x, column in zip(xs, columns, strict=True):
        if column not in named:
            continue
        ax.text(
            x,
            y + (NAME_PAD if above else -NAME_PAD),
            short(method, column),
            ha="center",
            va="bottom" if above else "top",
            fontsize=NAME_SIZE,
            fontweight="bold",
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
    (left, right), ys = _positions(cos2), (y, y - ROW_GAP)
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
    named = block.associated(dataset)
    _row(ax, left, ys[0], cos2.index, block.method, above=True, named=named[0])
    _row(ax, right, ys[1], cos2.columns, block.method, above=False, named=named[1])
    return ys


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


def size_key(ax) -> None:
    """The dots' sizes keyed in one line inside the border's top, centred
    between it and the top of PB's names (a name's height over the first row,
    the same in every panel), toward the left: named (an association at q <
    `common.SIG_ALPHA`) and not, in grey, as Figure 5G keys its dots."""
    y0, y1 = ax.get_ylim()
    per_inch = (y1 - y0) / (ax.bbox.height / ax.figure.dpi)
    names_top = NAME_PAD + NAME_SIZE / 72 * per_inch
    handles = [
        ax.scatter([], [], s=size, color=KEY_COLOR, lw=0) for size in (NAMED_DOT, DOT)
    ]
    ax.legend(
        loc="center left",
        bbox_to_anchor=(KEY_X, (TOP + names_top) / 2),
        bbox_transform=blended_transform_factory(ax.transAxes, ax.transData),
        handles=handles,
        labels=[f"q < {common.SIG_ALPHA}", f"q ≥ {common.SIG_ALPHA}"],
        ncols=2,
        fontsize=KEY_SIZE,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0,
        frameon=False,
    )


def coarse_to_fine(ax, dataset: str, method_names: bool, key: bool = False) -> None:
    """Each method's left fit's LR columns over its right fit's (module
    docstring); `method_names` names the methods left of the rows; `key` keys
    the dot sizes inside the panel (`size_key`)."""
    pairs = lr_matrix(dataset, "cccRISE").index
    tables = {b: b.cos2(pairs) for b in blocks(dataset)}
    bottom = -BLOCK_STEP * (len(tables) - 1) - ROW_GAP
    ax.set(xlim=(-X_PAD, 1 + X_PAD), ylim=(bottom - BORDER_DROP, TOP))
    common.source_data(
        f"lr_cos2_between_fits_{dataset}",
        pd.concat([_cos2_rows(b, t) for b, t in tables.items()], ignore_index=True),
        f"LR cos^2 on {DATASET_LABELS[dataset]} between each comparator's coarse and "
        "fine LR columns, and between CCC-RISE's LR columns at a reduced RISE rank and "
        "the reported fit's, on the reported fit's LR pairs; a line is drawn at "
        f"{FLOW_FROM} or more.",
    )
    common.source_data(
        f"named_column_associations_{dataset}",
        pd.concat([associated(dataset), reduced_associated(dataset)])[
            ["Method", "Labelling", "Component", "Association", "EtaSquared", "PAdj"]
        ].rename(columns={"EtaSquared": "Eta squared", "PAdj": "q"}),
        f"The associations at q < {common.SIG_ALPHA} of the columns named (PB and "
        "STAC columns by labelling; CCC-RISE components of the reported fit, "
        "labelling 'coarse', and of the reduced-RISE fit, labelling 'reduced'; "
        "q is FDR over each fit's associations and components).",
    )
    rows = {
        block: _block(ax, dataset, block, cos2, -BLOCK_STEP * i)
        for i, (block, cos2) in enumerate(tables.items())
    }
    _row_labels(ax, rows, method_names)
    common.data_border(ax, bottom - BORDER_DROP, TOP)
    if key:
        size_key(ax)
    cohort_bar(ax, dataset, -X_PAD, 1 + X_PAD)
