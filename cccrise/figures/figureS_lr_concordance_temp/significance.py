"""Are the pairs one method calls significant also significant in the other?

A pair is significant in a column when its |loading| is at least
`SIGNIFICANT` x the column's largest. Per method (own-rank fits on the
coarse annotation), each pair's score is its largest |loading| / column max
over the method's columns. Every pair in the top `TOP_K` of any column of
cccRISE or the comparator is drawn at (cccRISE score, comparator score); the
quadrants at `SIGNIFICANT` split them into significant in both, in cccRISE
only, in the comparator only and in neither. Pairs significant in cccRISE
are coloured by its component; pairs significant in neither are unlabelled."""

import numpy as np
import pandas as pd
import seaborn as sns

from .. import style
from ..figureS_lr_concordance._common import SHORT, load
from .missed import _spread

SIGNIFICANT = 0.4
TOP_K = 10
LIM = (-0.04, 1.06)
LABEL_GAP = 0.05
LEADER_SLOPE = 0.3
REGIONS = {  # (x significant, y significant) -> shade
    (True, True): style.COLORS["highlight"],
    (True, False): style.COLORS["up"],
    (False, True): style.COLORS["up"],
}


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


def _shade(ax):
    edges = {True: (SIGNIFICANT, LIM[1]), False: (LIM[0], SIGNIFICANT)}
    for (sx, sy), color in REGIONS.items():
        ax.fill_between(edges[sx], *edges[sy], color=color, alpha=0.1, lw=0, zorder=0)
    line = {"ls": style.LINE_STYLE_REFERENCE, "color": style.COLORS["reference"]}
    ax.axvline(SIGNIFICANT, lw=0.5, **line)
    ax.axhline(SIGNIFICANT, lw=0.5, **line)


def _counts(ax, table, short):
    """Pairs per region, in the region's empty corner next to the cuts."""
    sx, sy = table["x"] >= SIGNIFICANT, table["y"] >= SIGNIFICANT
    for (x, y, ha, va), n, name, key in (
        (
            (0.98, SIGNIFICANT + 0.02, "right", "bottom"),
            (sx & sy).sum(),
            "both",
            (True, True),
        ),
        (
            (0.98, SIGNIFICANT - 0.02, "right", "top"),
            (sx & ~sy).sum(),
            "cccRISE only",
            (True, False),
        ),
        (
            (0.0, SIGNIFICANT + 0.02, "left", "bottom"),
            (~sx & sy).sum(),
            f"{short} only",
            (False, True),
        ),
        (
            (0.0, SIGNIFICANT - 0.02, "left", "top"),
            (~sx & ~sy).sum(),
            "neither",
            (False, False),
        ),
    ):
        ax.text(
            x,
            y,
            f"{name}: {n}",
            ha=ha,
            va=va,
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold",
            color=REGIONS.get(key, style.COLORS["reference"]),
        )


def _label_text(pair, row, short):
    where = [f"C{int(row.component_x)}"] if row.x >= SIGNIFICANT else []
    if row.y >= SIGNIFICANT:
        where.append(f"{short} {int(row.component_y)}")
    return f"{pair.replace('^', '–')} ({' · '.join(where)})"


def _labels(ax, table, colors, short):
    """A label column right of the axes. Each label aims for where its leader
    would meet the column at a common slope (up from points low in the plot,
    down from those high in it), so leaders rarely cross."""
    reach = (LIM[1] - table["x"]) * LEADER_SLOPE  # labelled pairs only, below
    aim = table["y"] + np.where(table["y"] < 0.5, reach, -reach)
    rows = table.assign(aim=aim)
    rows = rows[(rows["x"] >= SIGNIFICANT) | (rows["y"] >= SIGNIFICANT)]
    rows = rows.sort_values("aim", ascending=False)
    tops = _spread(
        LIM[1] - rows["aim"].to_numpy(float), LABEL_GAP, 0.0, LIM[1] - LIM[0]
    )
    for top, (pair, row) in zip(tops, rows.iterrows(), strict=True):
        ax.annotate(
            _label_text(pair, row, short),
            (row.x, row.y),
            xytext=(1.04, LIM[1] - top),
            textcoords=("axes fraction", "data"),
            va="center",
            color=colors[pair],
            fontsize=style.FONT_SIZE_DENSE - 0.5,
            arrowprops={"arrowstyle": "-", "lw": 0.3, "color": style.COLORS["muted"]},
        )


def plot(fig, method):
    short = SHORT[method]
    table = scores(method)
    palette = sns.color_palette("Dark2", 5)
    colors = pd.Series(
        [
            palette[c - 1]
            if x >= SIGNIFICANT
            else style.COLORS["ink" if y >= SIGNIFICANT else "muted"]
            for c, x, y in zip(
                table["component_x"], table["x"], table["y"], strict=True
            )
        ],
        index=table.index,
    )
    ax = fig.subplots()
    _shade(ax)
    ax.scatter(
        table["x"],
        table["y"],
        s=style.POINT_SIZE_FOREGROUND,
        c=list(colors),
        edgecolor="white",
        lw=0.4,
        zorder=2,
    )
    _counts(ax, table, short)
    _labels(ax, table, colors, short)
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_aspect("equal")
    ax.set_xlabel("cccRISE |loading| / component max")
    ax.set_ylabel(f"{short} |loading| / component max")
    ax.set_title(f"cccRISE vs {method} (coarse)", fontsize=style.FONT_SIZE_TITLE)
    ax.grid(False)
