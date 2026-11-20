"""NicheNet ligand-weighted target score of each CCC-RISE component and each PB
and STAC LR column, against rankings with shuffled cells, ALAD and IBD cohorts.

Figure 5G: Figure 2's NicheNet test on CCC-RISE and every PB and STAC LR
column (``cccrise-comparator-nichenet``), ALAD beside IBD, laid out as the
depth-tracking panel lays out its fits (`_depth`): each column's
ligand-weighted target score above rankings with shuffled cells, filled and
named where q < 0.05. TC2C, whose columns track depth, is left out."""

import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from .. import common, style
from ._common import DATASET_LABELS, METHOD_COLORS, load
from ._data import LABELLINGS, short
from ._depth import cohort_bars

AXES = ("fig5_nichenet",)
#: CCC-RISE, then each fit but TC2C's (which tracks depth, `_depth`).
SLOTS = [("cccRISE", "coarse", "CCC-RISE")] + [
    (m, lab, f"{m} {lab[0].upper()}") for m in ("PB", "STAC") for lab in LABELLINGS
]
SCORE_LABEL = "NicheNet target score\nabove shuffled cells"
#: Cohort bars below the axes (mm), clear of the rotated fit names.
BAR_DROP = 9.5


def _strip(ax, x: float, g: pd.DataFrame, method: str, lab: str, rng) -> tuple:
    """One fit's columns at `x`: the q < 0.05 ones' (name, x, y), and every
    column's (x, y)."""
    xs = x + rng.uniform(-0.2, 0.2, len(g))
    hit = (g["q_cell"] < common.SIG_ALPHA).to_numpy()
    color = METHOD_COLORS[method]
    effect = g["Effect"].to_numpy()
    ax.scatter(
        xs[~hit],
        effect[~hit],
        s=8,
        facecolor="white",
        edgecolor=color,
        lw=0.6,
        zorder=3,
    )
    ax.scatter(xs[hit], effect[hit], s=11, color=color, lw=0, zorder=4)
    suffix = "" if method == "cccRISE" else f" {lab[0].upper()}"
    named = [
        (f"{short(method, int(c))}{suffix}", px, py)
        for c, px, py, h in zip(g["Component"], xs, effect, hit, strict=True)
        if h
    ]
    return named, list(zip(xs, effect, strict=True))


#: A name's candidate offsets from its point (points), nearest first: right
#: or left of it, level or stepping away up and down.
OFFSETS = sorted(
    ((side * dx, dy) for dx in (4, 9) for dy in range(-30, 31, 3) for side in (1, -1)),
    key=lambda o: (abs(o[0]) + abs(o[1]), -o[0]),
)
#: Names set by hand (offset from their point, points), moved after the
#: rest are placed so no other name moves.
FIXED = {"P4 C": (7, 9)}
#: A dot's radius (points) with room to spare, for the collision test.
DOT_RADIUS = 2.4


def _box(anchor, size, dx: float) -> Bbox:
    """A name's display box: `size` (w, h) left of or right of `anchor`,
    vertically centred on it."""
    (ax_, ay), (w, h) = anchor, size
    x0 = ax_ if dx > 0 else ax_ - w
    return Bbox.from_bounds(x0, ay - h / 2, w, h)


def _clashes(box: Bbox, dots: np.ndarray, radius: float, placed: list, frame) -> int:
    """How many dots and placed names `box` (display) touches; outside
    `frame` counts as many."""
    if not (frame.x0 <= box.x0 and box.x1 <= frame.x1):
        return 100
    if not (frame.y0 <= box.y0 and box.y1 <= frame.y1):
        return 100
    near = (
        (dots[:, 0] > box.x0 - radius)
        & (dots[:, 0] < box.x1 + radius)
        & (dots[:, 1] > box.y0 - radius)
        & (dots[:, 1] < box.y1 + radius)
    )
    return int(near.sum()) + sum(box.overlaps(b) for b in placed)


def _place(ax, text, point, dots, placed, renderer) -> None:
    """Move `text` (an annotation of `point`, display) to the first of
    `OFFSETS` it clashes with nothing at (`_clashes`); if none, the one it
    clashes with least."""
    scale = ax.figure.dpi / 72
    extent = text.get_window_extent(renderer)
    size = (extent.width, extent.height)

    def boxed(offset):
        return _box(
            (point[0] + offset[0] * scale, point[1] + offset[1] * scale),
            size,
            offset[0],
        )

    def clashes(offset):
        return _clashes(boxed(offset), dots, DOT_RADIUS * scale, placed, ax.bbox)

    best = next((o for o in OFFSETS if clashes(o) == 0), None)
    best = best or min(OFFSETS, key=clashes)
    text.xyann = best
    text.set_ha("left" if best[0] > 0 else "right")
    placed.append(boxed(best))


def _name_hits(ax, hits: list, points: list) -> None:
    """Each q < 0.05 column named beside its point, clear of every dot and
    name (`_place`), on a leader line from it."""
    renderer = ax.figure.canvas.get_renderer()
    dots = ax.transData.transform(np.array(points))
    placed: list = []
    texts = {}
    for name, x, y in sorted(hits, key=lambda h: -h[2]):
        text = ax.annotate(
            name,
            (x, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=style.FONT_SIZE_DENSE - 1.2,
            zorder=6,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.4,
                "color": "0.4",
                "shrinkA": 0,
                "shrinkB": 1,
            },
        )
        _place(ax, text, ax.transData.transform((x, y)), dots, placed, renderer)
        texts[name] = text
    for name, offset in FIXED.items():
        if name in texts:
            texts[name].xyann = offset
            texts[name].set_ha("left")


def _key(ax) -> None:
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color="0.4", ms=3.2, label="q < 0.05"),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                mfc="white",
                mec="0.4",
                ms=3.2,
                label="q ≥ 0.05",
            ),
        ],
        loc="lower left",
        fontsize=style.FONT_SIZE_DENSE - 1,
        handletextpad=0.2,
        borderaxespad=0.2,
        labelspacing=0.2,
        frameon=False,
    )


def plot(axs):
    (ax,) = axs
    rng = np.random.default_rng(2)
    x, positions, names, spans, frames, hits, points = 0.0, [], [], {}, [], [], []
    for dataset, label in DATASET_LABELS.items():
        t = load("nichenet", dataset=dataset)
        t = t.assign(Effect=t["weighted_auroc"] - t["null_mean_cell"], Cohort=label)
        frames.append(t)
        start = x
        for method, lab, name in SLOTS:
            fit = t[(t["Method"] == method) & (t["Labelling"] == lab)]
            named, dots = _strip(ax, x, fit, method, lab, rng)
            hits += named
            points += dots
            positions.append(x)
            names.append(name)
            x += 1
        spans[dataset] = (start, x - 1)
        x += 0.7
    t = pd.concat(frames, ignore_index=True)
    common.source_data(
        "nichenet_cell_shuffle_by_column",
        t[
            [
                "Cohort",
                "Method",
                "Labelling",
                "Component",
                "Rule",
                "High cells",
                "weighted_auroc",
                "null_mean_cell",
                "Effect",
                "p_cell",
                "q_cell",
            ]
        ],
        "Each column's ligand-weighted mean target score, its mean over rankings with "
        "the high and low labels shuffled, their difference (Effect), and the "
        "one-sided permutation p, Benjamini-Hochberg adjusted within the fit (q).",
    )
    shown = pd.DataFrame([s[:2] for s in SLOTS], columns=["Method", "Labelling"])
    common.source_data(
        "nichenet_cell_shuffle_null_by_column",
        pd.concat(
            load("nichenet_null", dataset=d).assign(Cohort=label).merge(shown)
            for d, label in DATASET_LABELS.items()
        )[
            [
                "Cohort",
                "Method",
                "Labelling",
                "Component",
                "Replicate",
                "weighted_auroc",
            ]
        ],
        "Each column's ligand-weighted mean target score on every ranking with the "
        "high and low labels shuffled: the null behind its mean and p.",
    )
    ax.axhline(0, color=style.COLORS["reference"], ls="--", lw=0.6, zorder=1)
    ax.set_xticks(
        positions,
        names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE - 0.7,
    )
    ax.set_xlim(-0.6, x - 0.7 + 0.6 - 1)
    ax.set_ylabel(SCORE_LABEL, fontsize=style.FONT_SIZE_DENSE)
    top = max(t["Effect"].max(), 0)
    ax.set_ylim(top=top + 0.03)
    ax.set_xlim(right=positions[-1] + 1.25)  # room for the last strip's names
    _name_hits(ax, hits, points)
    _key(ax)
    cohort_bars(ax, spans, BAR_DROP)


def makeFigure():
    return common.standalone_fixed(__name__)
