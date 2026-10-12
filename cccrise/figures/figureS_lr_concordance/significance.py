"""Are the pairs one method calls significant also significant in the other?

A pair is significant in a column when its |loading| is at least
`SIGNIFICANT` x the column's largest. Per method (own-rank fits; rows: coarse
and fine annotation, columns: comparators), each pair's score is its largest
|loading| / column max over the method's columns. Every pair in the top
`TOP_K` of any column of cccRISE or the comparator fit is drawn at (cccRISE
score, comparator score); the
solid lines at `SIGNIFICANT` split them into significant in both (grey), in
cccRISE only (green), in the comparator only (red) and in neither. The dashed
line at `C3_FLAG` only draws attention to cccRISE's C3 pairs between it and
`SIGNIFICANT` (drawn in C3's colour, not counted as significant). Points
are coloured by cccRISE component (comparator-only pairs in black unless
flagged C3); pairs significant in neither are drawn small and grey.
`membership` lists the pairs across all six fits (`combined`). Points are
jittered by up to
`JITTER` (never across a threshold) so pairs at the same scores, e.g. two
columns' top pair at (1, 1), stay visible."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .. import common, style
from ._common import ANNOTATIONS, METHODS, SHORT, load

SIGNIFICANT = 0.4
C3_FLAG = 0.2
TOP_K = 10
JITTER = 0.02  # axis units, uniform on each axis
ALPHA = 0.85
LIM = (-0.04, 1.06)
FIGURE_SIZE = (130 * style.MM, 100 * style.MM)
FITS = [(method, annotation) for annotation in ANNOTATIONS for method in METHODS]
REGIONS = {  # region -> (x significant, y significant)
    "cccRISE Only": (True, False),
    "Shared": (True, True),
    "Comparator Only": (False, True),
}
SHADES = {
    "cccRISE Only": to_rgba(style.COLORS["highlight"], 0.13),
    "Shared": "0.9",
    "Comparator Only": to_rgba(style.COLORS["up"], 0.12),
}


def scores(
    method: str, annotation: str = "coarse", ref: str = "cccRISE"
) -> pd.DataFrame:
    """Per pair in the top `TOP_K` of any column of `ref` or `method` (on
    `annotation`): each method's best |loading| / column max (`x` for `ref`,
    `y` for `method`) and the column it comes from."""
    table = load("profiles")
    table = table[table["annotation"] == annotation]
    best = table.loc[table.groupby(["method", "lr_pair"])["rel_max"].idxmax()]
    best = best.set_index(["method", "lr_pair"])[["rel_max", "component"]]
    out = best.loc[ref].join(best.loc[method], lsuffix="_x", rsuffix="_y")
    out = out.rename(columns={"rel_max_x": "x", "rel_max_y": "y"})
    top = table[table["method"].isin([ref, method]) & (table["lr_rank"] <= TOP_K)]
    return out.loc[sorted(set(top["lr_pair"]))]


def significant(table: pd.DataFrame) -> pd.DataFrame:
    """Each pair's significance in cccRISE (`sx`) and the comparator (`sy`),
    the region it falls in (None: neither) and whether it is a C3 pair
    flagged between `C3_FLAG` and `SIGNIFICANT`."""
    sx = table["x"] >= SIGNIFICANT
    sy = table["y"] >= SIGNIFICANT
    return table.assign(
        sx=sx,
        sy=sy,
        region=_regions(sx, sy),
        c3_flag=(table["component_x"] == 3) & (table["x"] >= C3_FLAG) & ~sx,
    )


def _regions(sx, sy):
    lookup = {flags: name for name, flags in REGIONS.items()}
    return [lookup.get(flags) for flags in zip(sx, sy, strict=True)]


def combined() -> pd.DataFrame:
    """Every pair from the comparator fits' `significant` tables, once, with
    its cccRISE score (`x`, `component_x`, `sx`, `c3_flag`), its best score
    over each comparator's coarse and fine fits (one column per short name)
    and its region across all of them (`sy`: significant in any comparator
    fit)."""
    tables = pd.concat(
        {fit: significant(scores(*fit)) for fit in FITS},
        names=["method", "annotation", "lr_pair"],
    ).reset_index()
    pairs = tables.groupby("lr_pair").agg(
        x=("x", "first"),
        component_x=("component_x", "first"),
        sx=("sx", "first"),
        c3_flag=("c3_flag", "first"),
        y=("y", "max"),
        sy=("sy", "any"),
    )
    per_method = tables.pivot_table(
        index="lr_pair", columns="method", values="y", aggfunc="max"
    ).rename(columns=SHORT)
    return pairs.join(per_method).assign(region=_regions(pairs["sx"], pairs["sy"]))


def shade(ax, c3=True, shades=None):
    """The regions' fills (`SHADES` unless given) and threshold lines (with
    `c3`, cccRISE's C3 flag line too)."""
    shades = shades or SHADES
    for name, (sx, sy) in REGIONS.items():
        x0, x1 = (SIGNIFICANT, LIM[1]) if sx else (LIM[0], SIGNIFICANT)
        y0, y1 = (SIGNIFICANT, LIM[1]) if sy else (LIM[0], SIGNIFICANT)
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, lw=0, color=shades[name], zorder=0)
        )
    line = {"color": "black", "lw": style.LINE_WIDTH_THIN, "zorder": 1}
    ax.axvline(SIGNIFICANT, **line)
    ax.axhline(SIGNIFICANT, **line)
    if c3:
        ax.axvline(C3_FLAG, ls=style.LINE_STYLE_REFERENCE, **line)


def _colors(table, palette):
    """Listed points by cccRISE component (comparator-only in black, but
    flagged C3 pairs in C3's colour); the rest grey."""
    return pd.Series(
        [
            palette[c - 1] if sx or flag else "black" if sy else style.COLORS["muted"]
            for c, sx, sy, flag in zip(
                table["component_x"],
                table["sx"],
                table["sy"],
                table["c3_flag"],
                strict=True,
            )
        ],
        index=table.index,
    )


def _jitter(values, thresholds, rng):
    """`values` moved by up to `JITTER`; a move across one of `thresholds` is
    mirrored back onto the value's own side of it."""
    moved = values + rng.uniform(-JITTER, JITTER, len(values))
    for cut in thresholds:
        crossed = (values >= cut) != (moved >= cut)
        moved = np.where(crossed, 2 * cut - moved, moved)
    return moved


def scatter(ax, table, colors):
    rng = np.random.default_rng(0)
    x = _jitter(table["x"].to_numpy(), (C3_FLAG, SIGNIFICANT), rng)
    y = _jitter(table["y"].to_numpy(), (SIGNIFICANT,), rng)
    big = (table["region"].notna() | table["c3_flag"]).to_numpy()
    for keep, size in ((~big, style.POINT_SIZE), (big, style.POINT_SIZE_FOREGROUND)):
        ax.scatter(
            x[keep], y[keep], s=size, c=list(colors[keep]), alpha=ALPHA,
            edgecolor="white", lw=0.4, zorder=2,
        )  # fmt: skip
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_aspect("equal")
    ax.grid(False)


def _n_components() -> int:
    profiles = load("profiles")
    return int(profiles.loc[profiles["method"] == "cccRISE", "component"].max())


def legend(fig):
    palette = sns.color_palette("Dark2", 8)
    marker = {"marker": "o", "ls": "", "markersize": 4, "mec": "white", "mew": 0.4}
    handles = [
        Line2D([], [], color=palette[c - 1], label=f"C{c}", **marker)
        for c in range(1, _n_components() + 1)
    ]
    handles.append(Line2D([], [], color="black", label="comparator", **marker))
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=style.FONT_SIZE_DENSE,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.2,
    )


def _panel(ax, method, annotation):
    table = significant(scores(method, annotation))
    shade(ax)
    scatter(ax, table, _colors(table, sns.color_palette("Dark2", 8)))
    ax.set_title(f"{method} ({annotation})", fontsize=style.FONT_SIZE_TITLE)


def plot(fig):
    """Rows: annotation; columns: comparator; below: one shared legend."""
    axes = fig.subplots(
        len(ANNOTATIONS), len(METHODS), sharex=True, sharey=True, squeeze=False
    )
    for ax, (method, annotation) in zip(axes.ravel(), FITS, strict=True):
        _panel(ax, method, annotation)
    for ax in axes[-1]:
        ax.set_xlabel("cccRISE |loading| / max")
    for ax in axes[:, 0]:
        ax.set_ylabel("comparator |loading| / max")
    legend(fig)


def makeFigure():
    return common.standalone(plot, FIGURE_SIZE)
