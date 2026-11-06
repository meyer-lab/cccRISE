"""How Figure 5 lays out components of every fit on one axis: cccRISE once
(it does not depend on the labels), then each comparator's coarse fit left
of its fine one, with C/F ticks and the method's name below; and how it
labels points without overlap."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
NULL_HANDLE = Line2D(
    [],
    [],
    marker="_",
    ls="",
    color=style.COLORS["reference"],
    markersize=6,
    label="stricter null, 95th pct.",
)
UNTESTABLE_HANDLE = Line2D(
    [],
    [],
    marker="x",
    ls="",
    color="0.35",
    markersize=3.5,
    mew=0.7,
    label="no gene-disjoint pairs",
)
#: Where `UNTESTABLE_HANDLE`'s x sits on a co-variation axis.
UNTESTABLE_Y = 0.04


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


def fit_axis(ax, slots, method_offset=-12, labelled: bool = True) -> None:
    """C/F ticks at each (method, labelling, x) of `slots`, cccRISE named at
    its own tick, and each comparator's name below, centred on its fits;
    unless `labelled` is false, which leaves the bare ticks (an axis stacked
    above another that names them)."""
    if not labelled:
        ax.set_xticks([x for *_, x in slots], [])
        return
    ax.set_xticks(
        [x for *_, x in slots],
        [_tick_label(m, lab) for m, lab, _ in slots],
        fontsize=style.FONT_SIZE_DENSE,
    )
    for label, (m, *_) in zip(ax.get_xticklabels(), slots, strict=True):
        label.set_fontsize(style.FONT_SIZE_DENSE - (1 if m == "cccRISE" else 0))
    _method_names(ax, slots, method_offset)


def _method_names(ax, slots, offset) -> None:
    """Each comparator's name `offset` points below the axis, centred on
    its fits."""
    names = {}
    for m, _, x in slots:
        if m != "cccRISE":
            names.setdefault(m, []).append(x)
    for m, xs in names.items():
        ax.annotate(
            METHOD_LABELS[m],
            (np.mean(xs), 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="top",
            annotation_clip=False,
        )


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
    `stagger`, a label next to (x within 1 of) the last one takes whichever
    of two heights sits further from it, so neighbours don't collide."""
    to_points = 72 / ax.figure.dpi
    previous = None  # the last label's x and baseline, in points
    for r in points.sort_values("X").itertuples():
        y = ax.transData.transform((r.X, r.Y))[1] * to_points
        offset = 3.5
        if stagger and previous is not None and r.X - previous[0] <= 1:
            offset = max((3.5, 9), key=lambda o: abs(y + o - previous[1]))
        previous = (r.X, y + offset)
        ax.annotate(
            r.Text,
            (r.X, r.Y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=size,
        )
