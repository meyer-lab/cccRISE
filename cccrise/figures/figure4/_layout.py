"""How Figure 4 lays out components of every fit on one axis: cccRISE once
(it does not depend on the labels), then each comparator's coarse fit left
of its fine one, with C/F ticks and the method's name below; and how it
labels points without overlap."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from .. import style
from ._common import COMPARATORS, METHOD_COLORS, METHOD_LABELS

LABELLING_MARKERS = {"coarse": "o", "fine": "^"}
#: Each fit's (method, labelling, x): cccRISE once, then each comparator's
#: coarse fit left of its fine one.
SLOTS = [("cccRISE", "coarse", 0.0)] + [
    (m, lab, i + 1 + (0.22 if lab == "fine" else -0.22))
    for i, m in enumerate(COMPARATORS)
    for lab in ("coarse", "fine")
]
LABELLING_HANDLES = [
    Line2D([], [], marker=mk, ls="", color="0.5", markersize=4, label=lab)
    for lab, mk in LABELLING_MARKERS.items()
]
NULL_HANDLE = Line2D(
    [],
    [],
    marker="_",
    ls="",
    color=style.COLORS["reference"],
    markersize=6,
    label="stricter null, 95th pct.",
)


def method_handles(methods, markersize=4) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=METHOD_COLORS[m],
            markersize=markersize,
            label=METHOD_LABELS[m],
        )
        for m in methods
    ]


def _tick_label(method: str, labelling: str) -> str:
    return METHOD_LABELS[method] if method == "cccRISE" else labelling[0].upper()


def fit_axis(ax, slots, method_offset=-12) -> None:
    """C/F ticks at each (method, labelling, x) of `slots`, cccRISE named at
    its own tick, and each comparator's name below, centred on its fits."""
    ax.set_xticks(
        [x for *_, x in slots],
        [_tick_label(m, lab) for m, lab, _ in slots],
        fontsize=style.FONT_SIZE_DENSE,
    )
    for label, (m, *_) in zip(ax.get_xticklabels(), slots, strict=True):
        label.set_fontsize(style.FONT_SIZE_DENSE - (1 if m == "cccRISE" else 0))
    names = {}
    for m, _, x in slots:
        if m != "cccRISE":
            names.setdefault(m, []).append(x)
    for m, xs in names.items():
        ax.annotate(
            METHOD_LABELS[m],
            (np.mean(xs), 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, method_offset),
            textcoords="offset points",
            ha="center",
            va="top",
            annotation_clip=False,
        )


def strips(ax, tables, y, filled=None) -> pd.DataFrame:
    """Each fit's components (`SLOTS`) at their fit's x, jittered, `y` from
    `tables[labelling]` (cccRISE's from the coarse one); hollow where
    `filled` is false. Returns the points drawn, with their X."""
    rng = np.random.default_rng(0)
    drawn = []
    for method, lab, centre in SLOTS:
        g = tables[lab][tables[lab]["Method"] == method]
        spread = 0.22 if method == "cccRISE" else 0.12
        xs = centre + rng.uniform(-spread, spread, len(g))
        color = METHOD_COLORS[method]
        fill = g[filled].to_numpy() if filled else np.ones(len(g), bool)
        ax.scatter(
            xs,
            g[y],
            marker=LABELLING_MARKERS[lab],
            s=style.POINT_SIZE * 1.2,
            facecolor=np.where(fill, color, "white"),
            edgecolor=color if filled else "white",
            lw=0.8 if filled else 0.4,
            zorder=3,
        )
        drawn.append(g.assign(X=xs, Labelling=lab))
    fit_axis(ax, SLOTS)
    return pd.concat(drawn, ignore_index=True)


def spread_slots(tables, slots=SLOTS) -> tuple[pd.DataFrame, list]:
    """Every component at its own x: each fit of `slots` in turn, in
    component order, with gaps between fits and wider ones between methods.
    Returns the components with X, and each fit's (method, labelling,
    centre) for `fit_axis`."""
    rows, centres, x, previous = [], [], 0.0, None
    for method, lab, _ in slots:
        g = tables[lab][tables[lab]["Method"] == method].sort_values("Component")
        if g.empty:
            continue
        x += 0 if previous is None else (2.0 if method != previous else 1.2)
        xs = x + np.arange(len(g))
        rows.append(g.assign(X=xs, Labelling=lab))
        centres.append((method, lab, xs.mean()))
        x, previous = xs[-1], method
    return pd.concat(rows, ignore_index=True), centres


def label_above(ax, points, stagger=False, size=style.FONT_SIZE_DENSE - 1.5) -> None:
    """Each row of `points` (X, Y, Text) labelled just above its point; with
    `stagger`, labels of neighbouring x alternate between two heights."""
    previous, raised = -np.inf, True
    for r in points.sort_values("X").itertuples():
        raised = stagger and (not raised if r.X - previous <= 1 else False)
        ax.annotate(
            r.Text,
            (r.X, r.Y),
            xytext=(0, 9 if raised else 3.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=size,
        )
        previous = r.X


def _obstacle_boxes(ax, obstacles) -> list[Bbox]:
    pad = 3 * ax.figure.dpi / 72
    xy = np.asarray(obstacles if obstacles is not None else np.empty((0, 2)), float)
    return [
        Bbox.from_extents(x - pad, y - pad, x + pad, y + pad)
        for x, y in ax.transData.transform(xy)
    ]


def _fits(ax, box, placed, overflow) -> bool:
    edge = ax.bbox
    inside = (
        edge.x0 - 2 <= box.x0
        and box.x1 <= edge.x1 + overflow
        and edge.y0 <= box.y0
        and box.y1 <= edge.y1 + 20
    )
    return inside and not any(box.overlaps(b) for b in placed)


def _place(ax, r, side, gap, spacing, size, renderer, placed, overflow):
    """`r`'s label on `side`, at the nearest height in `spacing` steps that
    `_fits`; None if none does."""
    for k in [0] + [k * sign for k in range(1, 30) for sign in (1, -1)]:
        label = ax.annotate(
            r.Text,
            (r.X, r.Y),
            xytext=(side * gap, k * spacing),
            textcoords="offset points",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=size,
            fontweight="bold" if r.Bold else "normal",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.4,
                "color": style.COLORS["reference"],
                "shrinkA": 0,
                "shrinkB": 2,
            },
        )
        label.update_positions(renderer)
        box = Text.get_window_extent(label, renderer).expanded(1.05, 1.15)
        if _fits(ax, box, placed, overflow):
            return label, box
        label.remove()
    return None


def label_sides(
    ax,
    points,
    gap=8,
    size=style.FONT_SIZE_DENSE - 1.5,
    spacing=3,
    overflow=0,
    obstacles=None,
) -> None:
    """Label each row of `points` (Method, Labelling, X, Y, Text, Bold) with
    a leader line: coarse comparator fits to the left, cccRISE and fine fits
    to the right (flipped where that would leave the axes' width, plus
    `overflow` points on the right), each at the nearest height clear of the
    labels already placed and of `obstacles` (data x, y). Labels are left
    out of the figure's layout."""
    ax.autoscale_view()
    renderer = ax.figure.canvas.get_renderer()
    placed = _obstacle_boxes(ax, obstacles)
    for r in points.sort_values("Y", ascending=False).itertuples():
        side = -1 if r.Method != "cccRISE" and r.Labelling == "coarse" else 1
        found = _place(
            ax, r, side, gap, spacing, size, renderer, placed, overflow
        ) or _place(ax, r, -side, gap, spacing, size, renderer, placed, overflow)
        if found:
            found[0].set_in_layout(False)
            placed.append(found[1])
