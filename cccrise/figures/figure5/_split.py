"""How a comparator holds a CCC-RISE component across its LR columns.

Each comparator is one fit: PB coarse, PB fine, STAC coarse, STAC fine (TC2C
left out). With the fit's unit LR columns U and the component's unit loading
c, the least squares weights b rebuild P c = U b, the best combination of the
fit's columns; ||P c||^2 is the component's containment (Figure 5B), and
column j's part of it is b_j (u_j . c).

A column is shown with a component (`related_columns`) when it bears on it
from either side:

- it rebuilds at least `REBUILT` of the component (its part of the
  containment times the containment), or
- at least `ON_COMPONENT` of the column's own squared loading sits on the
  component's carrying pairs (the column is about the component, even when it
  holds only a piece of it, as a column of one of its pairs).

A fit showing two or more columns also draws the sum, P c itself (`SUM`),
outlined and labelled with the containment."""

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ...covariation import CARRYING_CAP
from .. import common, style
from ._common import METHOD_COLORS, load
from ._data import LABELLINGS, short
from ._programs import _unit, carrying_pairs, lr_matrix
from ._recovery import (
    BLOCK_GAP,
    LINK,
    LOADING_LABEL,
    _cell,
    _pair_axes,
    colorbar_under,
    panel_title,
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


#: Signed panels' colormap, and the Σ outline (red, black on the diverging map).
DIVERGING = style.CMAPS["diverging"]
SIGNED_LABEL = "Loading / max |loading|"


def _paint(ax, x: float, y: float, value: float, signed: bool) -> None:
    if signed:
        color = DIVERGING((value + 1) / 2)
        ax.add_patch(Rectangle((x, y), 1, 1, fc=color, ec="black", lw=0.3))
    else:
        _cell(ax, x, y, abs(value))


def _scaled(loadings: pd.Series, pairs, signed: bool) -> pd.Series:
    """`pairs`' loadings over the column's heaviest |loading|, signed or not."""
    scale = loadings.abs().max()
    values = loadings.reindex(pairs).fillna(0) / scale if scale > 0 else 0 * loadings
    return values if signed else values.abs()


def _block(ax, x: float, rows, loadings: dict, signed: bool) -> list:
    """One column at `x`: each row's value from `loadings[program]` (a
    component's own column per program, or one comparator column for all)."""
    out = []
    for program, pair, y in rows:
        column = loadings[program]
        value = float(_scaled(column, [pair], signed).iloc[0])
        _paint(ax, x, y, value, signed)
        out.append((pair, value, program))
    return out


def _outline(ax, x: float, spans: dict, signed: bool) -> None:
    for y0, y1 in spans.values():
        ax.add_patch(
            Rectangle(
                (x, y0),
                1,
                y1 - y0,
                fc="none",
                ec="black" if signed else LINK,
                lw=1.1,
                zorder=6,
                clip_on=False,
            )
        )


def _signed_colorbar(ax, drop: float) -> None:
    cax = ax.inset_axes((0.17, -drop, 0.66, 0.04))
    ax.figure.colorbar(
        ScalarMappable(Normalize(-1, 1), DIVERGING), cax=cax, orientation="horizontal"
    )
    cax.set_xticks([-1, 0, 1])
    cax.tick_params(labelsize=style.FONT_SIZE_DENSE, length=1.5, pad=1)
    cax.set_xlabel(SIGNED_LABEL, fontsize=style.FONT_SIZE_DENSE, labelpad=0)


@dataclass
class _Heatmap:
    """What `pair_heatmap` builds up across its blocks: the next column's x,
    the column ticks and their labels, the blocks' heads, and the source rows."""

    ax: Any
    dataset: str
    programs: tuple
    rows: list
    spans: dict
    signed: bool
    x: float = 1 + BLOCK_GAP
    ticks: list = field(default_factory=lambda: [0.5])
    labels: list = field(default_factory=lambda: [""])
    heads: dict = field(default_factory=lambda: {"CCC-RISE": 0.5})
    values: list = field(default_factory=list)

    def column(self, loadings: dict, label: str) -> list:
        """One column at `x` (`_block`), ticked with `label`; its drawn cells."""
        drawn = _block(self.ax, self.x, self.rows, loadings, self.signed)
        self.x += 1
        self.ticks.append(self.x - 0.5)
        self.labels.append(label)
        return drawn


def _shares(parts: pd.DataFrame, programs) -> str:
    """One line per component, as the rows under CCC-RISE name them: the share
    of it the column rebuilds, or a dash."""
    held = dict(zip(parts["Program"], parts["Rebuilt"], strict=True))
    return "\n".join(f"{held[p]:.0%}" if p in held else "–" for p in programs)


def _related(h: _Heatmap, g: pd.DataFrame, method: str, lab: str) -> None:
    """A comparator's related columns, each ticked with its shares."""
    matrix = lr_matrix(h.dataset, method, lab)
    for column, parts in g.groupby("Column", sort=False):
        name = short(method, column)
        drawn = h.column(
            {p: matrix[column] for p in h.programs},
            f"{name}\n{_shares(parts, h.programs)}",
        )
        carries = ", ".join(f"C{q}" for q in parts["Program"])
        rebuilt = ", ".join(f"{r:.3f}" for r in parts["Rebuilt"])
        h.values += [
            (method, lab, name, pair, v, carries, rebuilt) for pair, v, _ in drawn
        ]


def _sum(h: _Heatmap, method: str, lab: str) -> None:
    """A comparator's best combination of all its columns, outlined, ticked
    with each component's containment."""
    sums = {p: projection(h.dataset, p, method, lab)[0] for p in h.programs}
    held = [projection(h.dataset, p, method, lab)[1].sum() for p in h.programs]
    _outline(h.ax, h.x, h.spans, h.signed)
    drawn = h.column(sums, f"{SUM}\n" + "\n".join(f"{v:.0%}" for v in held))
    h.values += [(method, lab, SUM, pair, v, f"C{p}", "") for pair, v, p in drawn]


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
    label = SIGNED_LABEL if h.signed else LOADING_LABEL
    common.source_data(
        f"component_pairs_in_related_columns_{h.dataset}",
        pd.DataFrame(
            h.values,
            columns=[
                "Method",
                "Labelling",
                "Column",
                "LR pair",
                label,
                "Component",
                "Share of component rebuilt",
            ],
        ),
        "Each component's carrying pairs in the component, in each PB and STAC "
        f"column rebuilding at least {REBUILT:.0%} of it or with at least "
        f"{ON_COMPONENT:.0%} of its own squared loading on the component's carrying "
        "pairs, and, where a comparator shows two or more columns, in its best "
        f"combination of all its columns (Column {SUM}, the projection whose squared "
        "norm is the containment); each column's "
        + ("signed loading" if h.signed else "|loading|")
        + " relative to its heaviest pair, and the share of each component's squared "
        "loading the column rebuilds.",
    )


def _heatmap_axes(h: _Heatmap) -> None:
    """Ticks and heads; the percentages' row names under the CCC-RISE column;
    one component unlabeled at the side (the title names it)."""
    h.labels[0] = "\n" + "\n".join(f"C{p} rebuilt" for p in h.programs)
    end = h.x - BLOCK_GAP
    ax = h.ax
    _pair_axes(ax, h.rows, h.spans, end, (h.ticks, h.labels, []), h.heads)
    for text in [t for t in ax.texts if t.get_text() == "CCC-RISE"]:
        text.set(x=1.0, ha="right")  # flush right of its one column
    if len(h.programs) == 1:
        _unlabel_side(ax, h.programs[0], end)
    ax.tick_params(axis="x", labelsize=style.FONT_SIZE_DENSE - 1.6)


def _unlabel_side(ax, program: int, end: float) -> None:
    """Drop `_pair_axes`' side label and bracket of the one component."""
    for text in [t for t in ax.texts if t.get_text() == f"C{program}"]:
        text.remove()
    for line in [ln for ln in ax.lines if set(ln.get_xdata()) == {end + 0.25}]:
        line.remove()


def pair_heatmap(
    ax,
    dataset: str,
    programs,
    title: str | None,
    drop: float = 0.24,
    signed: bool = False,
) -> None:
    """Each component's carrying pairs in the component and in its related
    columns of each comparator, then, where a comparator shows two or more
    columns, its best combination of all its columns (Σ, outlined); each column
    scaled to its heaviest pair, |loading| or `signed`. Under each column, the
    share of each component it rebuilds (Σ: the containment), one line per
    component, the lines named under the CCC-RISE column ("C4 rebuilt")."""
    rows, spans = _pair_rows(dataset, programs)
    h = _Heatmap(ax, dataset, tuple(programs), rows, spans, signed)
    cccrise = lr_matrix(dataset, "cccRISE")
    h.values = [
        ("cccRISE", "", f"C{p}", pair, v, f"C{p}", "")
        for pair, v, p in _block(
            ax, 0.0, rows, {p: cccrise[p] for p in programs}, signed
        )
    ]
    chosen = related_columns(dataset, programs)
    for method, lab in FITS:
        _fit_block(h, chosen, method, lab)
    _heatmap_source(h)
    _heatmap_axes(h)
    if title:
        panel_title(ax, title, pad=15)
    if signed:
        _signed_colorbar(ax, drop)
    else:
        colorbar_under(ax, LOADING_LABEL, drop=drop, x0=0.17, width=0.66)


def covariation_rows(dataset: str, programs) -> pd.DataFrame:
    """Each component, then its related columns comparator by comparator (as
    `pair_heatmap`), with their co-variation (`run_comparator_scores`)."""
    tables = {lab: load("covariation", lab, dataset) for lab in LABELLINGS}
    chosen = related_columns(dataset, programs)
    rows, x = [], 0.0
    for program in programs:
        t = tables["coarse"]
        picked = [
            ("cccRISE", t[(t.Method == "cccRISE") & (t.Component == program)], "")
        ]
        for r in chosen[chosen["Program"] == program].itertuples():
            s = tables[r.Labelling]
            picked.append(
                (
                    r.Method,
                    s[(s.Method == r.Method) & (s.Component == r.Column)],
                    r.Labelling,
                )
            )
        for method, row, lab in picked:
            component = int(row["Component"].iloc[0])
            suffix = f" {lab[0].upper()}" if lab else ""
            rows.append(
                row.assign(
                    X=x, Name=f"{short(method, component)}{suffix}", Labelling=lab
                )
            )
            x += 1
        x += 0.7
    return pd.concat(rows, ignore_index=True)


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


def covariation(axs, dataset: str, programs) -> None:
    """Each component's and related column's co-variation over all its carrying
    pairs, against both nulls' 95th percentiles (random dashed,
    expression-matched solid, keyed along the bottom); filled
    when above the expression-matched null."""
    (ax,) = axs
    t = covariation_rows(dataset, programs)
    common.source_data(
        f"component_and_related_column_covariation_{dataset}",
        t[
            [
                "Name",
                "Method",
                "Labelling",
                "PairsUsed",
                "GeneGroups",
                "Observed",
                "random_q95",
                "expression_matched_q95",
            ]
        ],
        "Mean pairwise rank correlation across samples of each component's or "
        "column's carrying pairs (the fewest carrying 90% of squared loading, at most "
        f"{CARRYING_CAP}), pairs sharing a gene merged into one (Gene "
        "groups of them, as Figure 3F), with the 95th percentiles of the random and "
        "of the expression-matched null, merged alike; NaN where fewer than two "
        "groups remain. A point is filled when above the expression-matched null.",
    )
    for _, r in t.iterrows():
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
        above = observed > r["expression_matched_q95"]
        ax.scatter(
            r.X,
            observed,
            s=14,
            lw=0.8,
            zorder=3,
            facecolor=color if above else "white",
            edgecolor=color,
        )
    ax.set(
        ylim=(0, 1),
        yticks=[0, 0.5, 1],
        ylabel="$\\bar{\\rho}$, shared\ngenes merged",
        xlim=(t["X"].min() - 0.7, t["X"].max() + 0.7),
    )
    ax.set_xticks(t["X"], t["Name"])
    ax.tick_params(axis="x", labelrotation=55, labelsize=style.FONT_SIZE_DENSE - 0.7)
    for label in ax.get_xticklabels():
        label.set(ha="right", rotation_mode="anchor")
    ax.legend(  # the nulls' 95th percentiles, flat along the bottom inside
        handles=[Line2D([], [], **tick) for tick in NULL_TICKS.values()],
        title=None,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=style.FONT_SIZE_DENSE - 1.2,
        handlelength=1.4,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.1,
        frameon=False,
    )
