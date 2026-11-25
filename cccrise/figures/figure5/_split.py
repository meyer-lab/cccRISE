"""How a comparator holds a CCC-RISE component across its LR columns.

Each comparator is one fit: PB coarse, PB fine, STAC coarse, STAC fine (TC2C
left out). With the fit's unit LR columns U and the component's unit loading
c, the least squares weights b rebuild P c = U b, the best combination of the
fit's columns; ||P c||^2 is the component's containment (Figure 5B), and
column j's part of it is b_j (u_j . c).

A column is shown with a component (`related_columns`) when it bears on it
from either side:

- it rebuilds at least `REBUILT` of the component (its part of the
  containment, b_j (u_j . c)), or
- at least `ON_COMPONENT` of the column's own squared loading sits on the
  component's carrying pairs (the column is about the component, even when it
  holds only a piece of it, as a column of one of its pairs).

A fit showing two or more columns also draws the sum, P c itself (`SUM`),
outlined in black."""

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .. import common, style
from ._common import METHOD_COLORS
from ._data import LABELLINGS, short
from ._programs import _unit, carrying_pairs, lr_matrix
from ._recovery import (
    BLOCK_GAP,
    LOADING_LABEL,
    _cell,
    _pair_axes,
)

#: The comparators, each labelling its own.
FITS = [(m, lab) for m in ("PB", "STAC") for lab in LABELLINGS]
#: A column rebuilding this share of a component is shown with it.
REBUILT = 0.10
#: A column with this share of its squared loading on a component's carrying
#: pairs is shown with it.
ON_COMPONENT = 0.25
SUM = "Σ"


def _pair_rows(dataset: str, programs) -> tuple[list, dict]:
    """Each component's carrying pairs, heaviest first, at their y, and each
    component's span of y."""
    rows, spans, y = [], {}, 0.0
    for program in programs:
        pairs = carrying_pairs(dataset)[program]  # heaviest first
        spans[program] = (y, y + len(pairs))
        rows += [(program, pair, y + i) for i, pair in enumerate(pairs)]
        y += len(pairs) + 0.7
    return rows, spans


@cache
def projection(dataset: str, program: int, method: str, labelling: str) -> tuple:
    """`program`'s best combination of the fit's columns: P c over LR pairs,
    and each column's part of ||P c||^2 (summing to the containment)."""
    programs = _unit(lr_matrix(dataset, "cccRISE"))
    c = programs[program].to_numpy()
    columns = _unit(
        lr_matrix(dataset, method, labelling).reindex(programs.index).fillna(0)
    )
    u = columns.to_numpy()
    b, *_ = np.linalg.lstsq(u, c, rcond=None)
    return (
        pd.Series(u @ b, index=programs.index),
        pd.Series(b * (u.T @ c), index=columns.columns.astype(int)),
    )


def on_component(dataset: str, program: int, method: str, labelling: str) -> pd.Series:
    """Each column's share of its squared loading on `program`'s carrying pairs."""
    squared = lr_matrix(dataset, method, labelling) ** 2
    held = squared.reindex(carrying_pairs(dataset)[program]).fillna(0).sum()
    return held / squared.sum()


def related_columns(dataset: str, programs) -> pd.DataFrame:
    """Every column shown with `programs` (module docstring): Program, Method,
    Labelling, Column, Rebuilt (share of the component it rebuilds),
    OnComponent, and Containment (the fit's, for the component). In `FITS`
    order, then component, then Rebuilt."""
    rows = []
    for fit, (method, lab) in enumerate(FITS):
        for rank, program in enumerate(programs):
            _, parts = projection(dataset, program, method, lab)
            on = on_component(dataset, program, method, lab)
            shown = parts.index[
                (parts >= REBUILT) | (on.reindex(parts.index) >= ON_COMPONENT)
            ]
            rows += [
                {
                    "Fit": fit,
                    "Rank": rank,
                    "Program": program,
                    "Method": method,
                    "Labelling": lab,
                    "Column": int(column),
                    "Rebuilt": float(parts[column]),
                    "OnComponent": float(on[column]),
                    "Containment": float(parts.sum()),
                }
                for column in shown
            ]
    t = pd.DataFrame(rows)
    return t.sort_values(["Fit", "Rank", "Rebuilt"], ascending=[True, True, False])


def _fit_name(method: str, lab: str) -> str:
    return f"{method} {lab[0].upper()}"


def _scaled(loadings: pd.Series, pairs) -> pd.Series:
    """`pairs`' |loadings| over the column's heaviest |loading|."""
    scale = loadings.abs().max()
    values = loadings.reindex(pairs).fillna(0) / scale if scale > 0 else 0 * loadings
    return values.abs()


def _block(ax, x: float, rows, loadings: dict) -> list:
    """One column at `x`: each row's value from `loadings[program]` (a
    component's own column per program, or one comparator column for all)."""
    out = []
    for program, pair, y in rows:
        value = float(_scaled(loadings[program], [pair]).iloc[0])
        _cell(ax, x, y, value)
        out.append((pair, value, program))
    return out


def _outline(ax, x: float, spans: dict) -> None:
    for y0, y1 in spans.values():
        ax.add_patch(
            Rectangle(
                (x, y0),
                1,
                y1 - y0,
                fc="none",
                ec="black",
                lw=1.1,
                zorder=6,
                clip_on=False,
            )
        )


@dataclass
class _Heatmap:
    """What `pair_heatmap` builds up across its blocks: the next column's x,
    the column ticks and their labels, the blocks' heads, and the source rows."""

    ax: Any
    dataset: str
    programs: tuple
    rows: list
    spans: dict
    x: float = 1 + BLOCK_GAP
    ticks: list = field(default_factory=lambda: [0.5])
    labels: list = field(default_factory=lambda: [""])
    heads: dict = field(default_factory=lambda: {"CCC-RISE": 0.5})
    values: list = field(default_factory=list)

    def column(self, loadings: dict, label: str) -> list:
        """One column at `x` (`_block`), ticked with `label`; its drawn cells."""
        drawn = _block(self.ax, self.x, self.rows, loadings)
        self.x += 1
        self.ticks.append(self.x - 0.5)
        self.labels.append(label)
        return drawn


def _related(h: _Heatmap, g: pd.DataFrame, method: str, lab: str) -> None:
    """A comparator's related columns, each ticked with its name."""
    matrix = lr_matrix(h.dataset, method, lab)
    for column, parts in g.groupby("Column", sort=False):
        name = short(method, column)
        # ticked with its number: the block's head names the fit
        drawn = h.column({p: matrix[column] for p in h.programs}, f"{column}")
        carries = ", ".join(f"C{q}" for q in parts["Program"])
        h.values += [(method, lab, name, pair, v, carries) for pair, v, _ in drawn]


def _sum(h: _Heatmap, method: str, lab: str) -> None:
    """A comparator's best combination of all its columns, outlined."""
    sums = {p: projection(h.dataset, p, method, lab)[0] for p in h.programs}
    _outline(h.ax, h.x, h.spans)
    drawn = h.column(sums, SUM)
    h.values += [(method, lab, SUM, pair, v, f"C{p}") for pair, v, p in drawn]


def _fit_block(h: _Heatmap, chosen: pd.DataFrame, method: str, lab: str) -> None:
    """One comparator's related columns, then its Σ where it has two or more."""
    g = chosen[(chosen["Method"] == method) & (chosen["Labelling"] == lab)]
    if g.empty:
        return
    start = h.x
    _related(h, g, method, lab)
    if g["Column"].nunique() > 1:
        _sum(h, method, lab)
    h.heads[_fit_name(method, lab)] = (start + h.x) / 2
    h.x += BLOCK_GAP


def _heatmap_source(h: _Heatmap) -> None:
    common.source_data(
        f"component_pairs_in_related_columns_{h.dataset}",
        pd.DataFrame(
            h.values,
            columns=[
                "Method",
                "Labelling",
                "Column",
                "LR pair",
                LOADING_LABEL,
                "Component",
            ],
        ),
        "Each component's carrying pairs in the component, in each PB and STAC "
        f"column reconstructing at least {REBUILT:.0%} of it or with at least "
        f"{ON_COMPONENT:.0%} of its own squared loading on the component's carrying "
        "pairs, and, where a comparator shows two or more columns, in its best "
        f"combination of all its columns (Column {SUM}, the projection whose squared "
        "norm is the containment); each column's |loading| relative to its "
        "heaviest pair.",
    )


def _heatmap_axes(h: _Heatmap) -> None:
    """Ticks and heads, the CCC-RISE head flush right of its one column; one
    component unlabeled at the side (the CCC-RISE column's tick names it)."""
    end = h.x - BLOCK_GAP
    ax = h.ax
    _pair_axes(ax, h.rows, h.spans, end, (h.ticks, h.labels, []), h.heads)
    for text in [t for t in ax.texts if t.get_text() == "CCC-RISE"]:
        text.set(x=1.0, ha="right")
    ax.tick_params(axis="x", labelsize=style.FONT_SIZE_LABEL)
    if len(h.programs) == 1:
        _unlabel_side(ax, h.programs[0], end)


def _unlabel_side(ax, program: int, end: float) -> None:
    """Drop `_pair_axes`' side label and bracket of the one component."""
    for text in [t for t in ax.texts if t.get_text() == f"C{program}"]:
        text.remove()
    for line in [ln for ln in ax.lines if set(ln.get_xdata()) == {end + 0.25}]:
        line.remove()


def pair_heatmap(ax, dataset: str, programs, name: str | None) -> None:
    """Each component's carrying pairs in the component and in its related
    columns of each comparator, then, where a comparator shows two or more
    columns, its best combination of all its columns (Σ, outlined); each column
    scaled to its heaviest |loading|, each ticked with its name, the
    component's column with `name` (e.g. 'ALAD C4'). No colorbar."""
    rows, spans = _pair_rows(dataset, programs)
    h = _Heatmap(ax, dataset, tuple(programs), rows, spans)
    cccrise = lr_matrix(dataset, "cccRISE")
    h.values = [
        ("cccRISE", "", f"C{p}", pair, v, f"C{p}")
        for pair, v, p in _block(ax, 0.0, rows, {p: cccrise[p] for p in programs})
    ]
    chosen = related_columns(dataset, programs)
    for method, lab in FITS:
        _fit_block(h, chosen, method, lab)
    _heatmap_source(h)
    h.labels[0] = name or ""
    _heatmap_axes(h)


#: Each null's 95th-percentile tick in the co-variation panels.
NULL_TICKS: dict[str, dict[str, Any]] = {
    "random": {"color": "0.6", "ls": (0, (2, 1.2)), "lw": 0.9, "label": "Random"},
    "expression_matched": {
        "color": "0.15",
        "ls": "-",
        "lw": 1.0,
        "label": "Expr.-matched",
    },
}


def _lollipop(ax, r) -> None:
    """One row's observed co-variation over its nulls' ticks, filled where its
    q is below `common.SIG_ALPHA`."""
    observed, color = r["Observed"], METHOD_COLORS[r.Method]
    bars = [r[f"{null}_q95"] for null in NULL_TICKS]
    ax.plot(
        [r.X, r.X],
        [min(observed, *bars), max(observed, *bars)],
        color="0.8",
        lw=0.6,
        zorder=1,
    )
    for null, tick in NULL_TICKS.items():
        ax.plot(
            [r.X - 0.3, r.X + 0.3],
            [r[f"{null}_q95"]] * 2,
            zorder=2,
            **{k: v for k, v in tick.items() if k != "label"},
        )
    hit = r["expression_matched_q"] < common.SIG_ALPHA
    ax.scatter(
        r.X,
        observed,
        s=14,
        lw=0.8,
        zorder=3,
        facecolor=color if hit else "white",
        edgecolor=color,
    )
