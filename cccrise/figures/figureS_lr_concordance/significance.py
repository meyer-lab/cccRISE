"""Are the pairs one method calls significant also significant in the other?

A pair is significant in a column when its |loading| is at least
`SIGNIFICANT` x the column's largest. Per method (own-rank fits on the
coarse annotation), each pair's score is its largest |loading| / column max
over the method's columns. Every pair in the top `TOP_K` of any column of
cccRISE or the comparator is drawn at (cccRISE score, comparator score); the
solid lines at `SIGNIFICANT` split them into significant in both (grey), in
cccRISE only (green), in the comparator only (red) and in neither. The dashed
line at `C3_FLAG` only draws attention to cccRISE's C3 pairs between it and
`SIGNIFICANT` (drawn in C3's colour, not counted as significant). Each
region's pairs are listed in a box beside the plot, coloured by their cccRISE
component (comparator-only pairs in black unless flagged C3); pairs significant in neither are
drawn small and grey and are not listed. Points are jittered by up to
`JITTER` (never across a threshold) so pairs at the same scores, e.g. two
columns' top pair at (1, 1), stay visible."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, ScaledTranslation

from .. import style
from ._common import SHORT, load

SIGNIFICANT = 0.4
C3_FLAG = 0.2
TOP_K = 10
JITTER = 0.02  # axis units, uniform on each axis
ALPHA = 0.85
LIM = (-0.04, 1.06)
FIGURE_SIZE = (150 * style.MM, 88 * style.MM)
REGIONS = {  # box title -> (x significant, y significant), top to bottom
    "cccRISE Only": (True, False),
    "Shared": (True, True),
    "Comparator Only": (False, True),
}
SHADES = {
    "cccRISE Only": to_rgba(style.COLORS["highlight"], 0.13),
    "Shared": "0.9",
    "Comparator Only": to_rgba(style.COLORS["up"], 0.12),
}
TWO_COLUMNS = 6  # a box with more pairs than this lists them in two columns
# Box layout, in points.
TITLE_SIZE = style.FONT_SIZE_DENSE
PAIR_SIZE = style.FONT_SIZE_DENSE - 0.5
ROW = 7.5  # pitch of the listed pairs
TITLE_ROW = 10  # title band, above the rule
PAD = 4  # inside the box, around the list
COLUMN_GAP = 10
BOX_GAP = 6  # between boxes


def scores(method: str) -> pd.DataFrame:
    """Per pair in the top `TOP_K` of any column of cccRISE or `method`: each
    method's best |loading| / column max (`x`, `y`) and the column it comes
    from."""
    table = load("profiles")
    table = table[table["annotation"] == "coarse"]
    best = table.loc[table.groupby(["method", "lr_pair"])["rel_max"].idxmax()]
    best = best.set_index(["method", "lr_pair"])[["rel_max", "component"]]
    out = best.loc["cccRISE"].join(best.loc[method], lsuffix="_x", rsuffix="_y")
    out = out.rename(columns={"rel_max_x": "x", "rel_max_y": "y"})
    top = table[table["method"].isin(["cccRISE", method]) & (table["lr_rank"] <= TOP_K)]
    return out.loc[sorted(set(top["lr_pair"]))]


def significant(table: pd.DataFrame) -> pd.DataFrame:
    """Each pair's significance in cccRISE (`sx`) and the comparator (`sy`),
    the region it falls in (None: neither) and whether it is a C3 pair
    flagged between `C3_FLAG` and `SIGNIFICANT`."""
    sx = table["x"] >= SIGNIFICANT
    sy = table["y"] >= SIGNIFICANT
    lookup = {flags: name for name, flags in REGIONS.items()}
    return table.assign(
        sx=sx,
        sy=sy,
        region=[lookup.get(flags) for flags in zip(sx, sy, strict=True)],
        c3_flag=(table["component_x"] == 3) & (table["x"] >= C3_FLAG) & ~sx,
    )


def _shade(ax):
    for name, (sx, sy) in REGIONS.items():
        x0, x1 = (SIGNIFICANT, LIM[1]) if sx else (LIM[0], SIGNIFICANT)
        y0, y1 = (SIGNIFICANT, LIM[1]) if sy else (LIM[0], SIGNIFICANT)
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, lw=0, color=SHADES[name], zorder=0)
        )
    line = {"color": "black", "lw": style.LINE_WIDTH_THIN, "zorder": 1}
    ax.axvline(SIGNIFICANT, **line)
    ax.axhline(SIGNIFICANT, **line)
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


def _scatter(ax, table, colors):
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
    ax.set_anchor("E")  # against the boxes
    ax.grid(False)


def _width(text, size, weight="normal"):
    """Width of `text` in points."""
    prop = FontProperties(size=size, weight=weight)
    return TextPath((0, 0), text, prop=prop).get_extents().width


def _entries(table, colors, short):
    """Per region: its title and its pairs as (label, colour) rows, in the
    pair's point colour (comparator-only pairs in black unless flagged C3)."""
    boxes = {}
    for name in REGIONS:
        rows = table[table["region"] == name]
        if name == "Comparator Only":
            rows = rows.sort_values("y", ascending=False)
        else:
            rows = rows.sort_values(["component_x", "x"], ascending=[True, False])
        labels = [
            pair.replace("^", "–")
            + ("" if name == "cccRISE Only" else f" ({short} {int(row.component_y)})")
            for pair, row in rows.iterrows()
        ]
        title = f"{name.replace('Comparator', short)} ({len(rows)})"
        listed = (
            [
                colors[pair] if flag else "black"
                for pair, flag in zip(rows.index, rows["c3_flag"], strict=True)
            ]
            if name == "Comparator Only"
            else list(colors[rows.index])
        )
        boxes[title] = list(zip(labels, listed, strict=True))
    return boxes


def _columns(entries):
    per = -(-len(entries) // 2) if len(entries) > TWO_COLUMNS else len(entries)
    return [entries[i : i + per] for i in range(0, len(entries), max(per, 1))]


def _box(ax, top, width, title, cols, common):
    """One box from `top` (points) down: centred title, rule, then the
    pairs in `cols`; returns the top of the next box."""
    rows = max((len(c) for c in cols), default=0)
    height = TITLE_ROW + PAD + ROW * max(rows, 1) + PAD / 2
    lw = style.LINE_WIDTH_THIN
    ax.add_patch(
        Rectangle((0, top), width, height, fill=False, ec="black", lw=lw, **common)
    )
    ax.text(
        width / 2, top + TITLE_ROW / 2 + 1, title, ha="center", va="center",
        fontsize=TITLE_SIZE, fontweight="bold", **common,
    )  # fmt: skip
    ax.add_line(
        Line2D([0, width], [top + TITLE_ROW + 1] * 2, color="black", lw=lw, **common)
    )
    x = PAD
    for col in cols:
        for i, (label, color) in enumerate(col):
            ax.text(
                x, top + TITLE_ROW + PAD + (i + 0.5) * ROW, label, va="center",
                fontsize=PAIR_SIZE, color=color, **common,
            )  # fmt: skip
        x += max(_width(label, PAIR_SIZE) for label, _ in col) + COLUMN_GAP
    return top + height + BOX_GAP


def _box_width(title, cols):
    widths = [max(_width(label, PAIR_SIZE) for label, _ in col) for col in cols]
    list_width = sum(widths) + COLUMN_GAP * (len(widths) - 1)
    return 2 * PAD + max(list_width, _width(title, TITLE_SIZE, "bold"))


def _boxes(ax, table, colors, short):
    """Sharp black boxes stacked from the top left of `ax`, laid out in
    points so each is as tall as its list; all share the widest's width."""
    ax.set_axis_off()
    # points, y down, from the axes' top-left corner
    at = (
        Affine2D().scale(1 / 72, -1 / 72)
        + ax.figure.dpi_scale_trans
        + ScaledTranslation(0, 1, ax.transAxes)
    )
    common = {"transform": at, "clip_on": False, "in_layout": False}
    boxes = {t: _columns(e) for t, e in _entries(table, colors, short).items()}
    width = max(_box_width(t, cols) for t, cols in boxes.items())
    top = 0.0
    for title, cols in boxes.items():
        top = _box(ax, top, width, title, cols, common)


def _legend(fig, palette, short):
    marker = {"marker": "o", "ls": "", "markersize": 4, "mec": "white", "mew": 0.4}
    handles = [
        Line2D([], [], color=palette[c - 1], label=f"C{c}", **marker)
        for c in range(1, 6)
    ]
    handles += [
        Line2D([], [], color="black", label=f"{short} only", **marker),
        Line2D([], [], color=style.COLORS["muted"], label="neither", **marker),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=style.FONT_SIZE_DENSE,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.2,
    )


def plot(fig, method):
    short = SHORT[method]
    table = significant(scores(method))
    palette = sns.color_palette("Dark2", 5)
    colors = _colors(table, palette)
    ax, side = fig.subplots(1, 2, width_ratios=(1, 0.72))
    _shade(ax)
    _scatter(ax, table, colors)
    ax.set_xlabel("cccRISE |loading| / component max")
    ax.set_ylabel(f"{short} |loading| / component max")
    ax.set_title(f"cccRISE vs {method}", fontsize=style.FONT_SIZE_TITLE)
    _boxes(side, table, colors, short)
    _legend(fig, palette, short)
