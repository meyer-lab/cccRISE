"""Figure 5E: Figure 2's NicheNet test on CCC-RISE and every PB and STAC LR
column (``cccrise-comparator-nichenet``), ALAD beside IBD, laid out as the
depth-tracking panel lays out its fits (`_depth`): each column's
ligand-weighted target score above rankings with shuffled cells, filled and
named where q < 0.05. TC2C, whose columns track depth, is left out."""

import numpy as np
import pandas as pd

from .. import common, style
from ._common import DATASET_LABELS, METHOD_COLORS, load
from ._data import LABELLINGS, short
from ._depth import cohort_bars

AXES = ("fig5_nichenet",)
#: CCC-RISE, then each fit but TC2C's (which tracks depth, `_depth`).
SLOTS = [("cccRISE", "coarse", "CCC-RISE")] + [
    (m, lab, f"{m} {lab[0].upper()}") for m in ("PB", "STAC") for lab in LABELLINGS
]
SCORE_LABEL = "Weighted target score\nabove shuffled cells"
#: Cohort bars below the axes (mm), clear of the rotated fit names.
BAR_DROP = 9.5


def _strip(ax, x: float, g: pd.DataFrame, method: str, lab: str, rng) -> list:
    """One fit's columns at `x`; the q < 0.05 ones' (name, x, y)."""
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
    return [
        (f"{short(method, int(c))}{suffix}", px, py)
        for c, px, py, h in zip(g["Component"], xs, effect, hit, strict=True)
        if h
    ]


def _spread(ys: list[float], gap: float) -> list[float]:
    """`ys` (highest first) moved down so each is at least `gap` below the last."""
    out: list[float] = []
    for y in ys:
        out.append(min(y, out[-1] - gap) if out else y)
    return out


def _name_hits(ax, hits: list, slots: list[float]) -> None:
    """Each q < 0.05 column named in a column right of its strip, names spread
    at least one text height apart, each on a leader line from its point."""
    height = ax.bbox.height / ax.figure.dpi
    low, high = ax.get_ylim()
    gap = 1.9 * style.MM / height * (high - low)  # one name's height, in data
    for slot in slots:
        mine = sorted(
            (h for h in hits if abs(h[1] - slot) <= 0.25), key=lambda h: -h[2]
        )
        tx = slot + 0.3
        for (name, x, y), ty in zip(
            mine, _spread([h[2] for h in mine], gap), strict=True
        ):
            ax.plot([x, tx - 0.04], [y, ty], color="0.4", lw=0.4, zorder=5)
            ax.text(
                tx,
                ty,
                name,
                ha="left",
                va="center",
                fontsize=style.FONT_SIZE_DENSE - 1.2,
                zorder=6,
            )


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
    x, positions, names, spans, frames, hits = 0.0, [], [], {}, [], []
    for dataset, label in DATASET_LABELS.items():
        t = load("nichenet", dataset=dataset)
        t = t.assign(Effect=t["weighted_auroc"] - t["null_mean_cell"], Cohort=label)
        frames.append(t)
        start = x
        for method, lab, name in SLOTS:
            fit = t[(t["Method"] == method) & (t["Labelling"] == lab)]
            hits += _strip(ax, x, fit, method, lab, rng)
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
    ax.set_xlim(right=positions[-1] + 0.95)  # room for the last strip's names
    _name_hits(ax, hits, positions)
    _key(ax)
    cohort_bars(ax, spans, BAR_DROP)


def makeFigure():
    return common.standalone_fixed(__name__)
