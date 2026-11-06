"""How each comparator column of Figure 4B gets its class: its value on each
rule, one point per column, coloured by the class the rules give it."""

import numpy as np
import pandas as pd

from ...comparators.scores import CONTAINED
from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS
from ._data import HOMOPHILIC
from ._programs import (
    BROAD_PAIRS,
    COLUMN_CLASSES,
    FITS,
    PART_SHARE,
    SINGLE_PAIRS,
    column_census,
)
from ._recovery import panel_title

#: Half the width a fit's columns spread over, and between labelled programs.
SPREAD, PROGRAM_STEP = 0.42, 0.6
#: Where leader lines end: the fit with the most programs one tier up.
LEADER_TOP, LEADER_TOP_HIGH = 1.12, 1.3
#: Cohort bar below the axes, and the room below it for the figure's key (mm).
BAR_DROP, KEY_ROOM = 8.5, 11
LABEL_SIZE = style.FONT_SIZE_DENSE - 1


def _jitter(n: int, x: float) -> np.ndarray:
    return x + np.random.default_rng(1).uniform(-0.22, 0.22, n)


def _cohort_bars(ax, spans: dict) -> None:
    height = ax.bbox.height / ax.figure.dpi
    drop = BAR_DROP * style.MM / height
    transform = ax.get_xaxis_transform()
    for dataset, (first, last) in spans.items():
        ax.plot(
            [first - 0.4, last + 0.4],
            [-drop, -drop],
            color=DATASET_COLORS[dataset],
            lw=2.5,
            transform=transform,
            clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            (first + last) / 2,
            -drop - style.MM / height,
            DATASET_LABELS[dataset],
            transform=transform,
            ha="center",
            va="top",
            fontsize=style.FONT_SIZE_DENSE,
        )


def key_room(ax) -> None:
    """Reserve `KEY_ROOM` under `ax`'s cohort labels: margins are measured from
    what an axes draws, and the key is the figure's."""
    height = ax.bbox.height / ax.figure.dpi
    ax.text(
        0.5,
        -(BAR_DROP + 4 + KEY_ROOM) * style.MM / height,
        "room",
        alpha=0,
        transform=ax.transAxes,
        fontsize=1,
    )


def _fit_axis(ax, positions: list, spans: dict, ylabel: str, title: str) -> None:
    ax.set_xticks(
        positions,
        [f"{m} {lab[0].upper()}" for _ in spans for m, lab in FITS],
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=style.FONT_SIZE_DENSE - 0.7,
    )
    ax.set_ylabel(ylabel)
    panel_title(ax, title)
    _cohort_bars(ax, spans)


def _source(name: str, value: str, description: str) -> None:
    t = pd.concat(column_census(d) for d in DATASET_LABELS)
    common.source_data(
        name,
        t[["Dataset", "Method", "Labelling", "Component", value, "Class"]],
        description,
    )


def _strip(ax, value: str, classes, place) -> tuple[list, dict]:
    """Every column of `classes` at its fit's x, `place(g, x)` giving each
    fit's points' x; each fit's x and each cohort's span of them."""
    x, positions, spans = 0.0, [], {}
    for dataset in DATASET_LABELS:
        t = column_census(dataset)
        t = t[t["Class"].isin(classes)]
        start = x
        for method, lab in FITS:
            g = t[(t["Method"] == method) & (t["Labelling"] == lab)]
            ax.scatter(
                place(g, x),
                g[value],
                s=9,
                lw=0,
                zorder=3,
                color=[COLUMN_CLASSES[c] for c in g["Class"]],
            )
            positions.append(x)
            x += 1
        spans[dataset] = (start, x - 1)
        x += 0.8
    return positions, spans


def _threshold(ax, *values: float) -> None:
    for value in values:
        ax.axhline(
            value,
            color=style.COLORS["reference"],
            ls=style.LINE_STYLE_REFERENCE,
            lw=0.6,
            zorder=1,
        )


def containment(ax) -> None:
    _source(
        "comparator_column_containment",
        "Containment",
        "Each comparator LR column's containment in cccRISE's LR span, and its class.",
    )
    positions, spans = _strip(
        ax, "Containment", COLUMN_CLASSES, lambda g, x: _jitter(len(g), x)
    )
    _threshold(ax, CONTAINED)
    ax.set_ylim(-0.04, 1.04)
    _fit_axis(ax, positions, spans, "LR containment", "cccRISE span containment")
    key_room(ax)


def breadth(ax) -> None:
    _source(
        "comparator_column_effective_pairs",
        "EffectiveLR",
        "Each comparator LR column's effective LR pairs, and its class.",
    )
    positions, spans = _strip(
        ax, "EffectiveLR", COLUMN_CLASSES, lambda g, x: _jitter(len(g), x)
    )
    _threshold(ax, SINGLE_PAIRS, BROAD_PAIRS)
    ax.set_yscale("log")
    _fit_axis(ax, positions, spans, "Effective LR pairs", "LR column breadth")


def homophily(ax) -> None:
    _source(
        "comparator_column_homophilic_share",
        "HomophilicShare",
        "Each comparator LR column's share of squared loading on homophilic pairs, "
        "for the columns the earlier rules leave.",
    )
    left = [
        k
        for k in COLUMN_CLASSES
        if k not in ("Same as a cccRISE program", "Part of a cccRISE program", "Broad")
    ]
    positions, spans = _strip(
        ax, "HomophilicShare", left, lambda g, x: _jitter(len(g), x)
    )
    _threshold(ax, HOMOPHILIC)
    ax.set_ylim(-0.04, 1.04)
    _fit_axis(ax, positions, spans, "Homophilic share", "Homophilic pairs")


def _spread(g: pd.DataFrame, x: float) -> np.ndarray:
    """Columns past `PART_SHARE` evenly across the fit, program by program;
    the rest jittered."""
    xs = _jitter(len(g), x)
    hit = (g["OnProgram"] >= PART_SHARE).to_numpy()
    if hit.sum() > 1:
        order = np.lexsort(
            (g["OnProgram"].to_numpy()[hit], g["Program"].to_numpy()[hit])
        )
        placed = np.empty(hit.sum())
        placed[order] = np.linspace(-SPREAD, SPREAD, hit.sum())
        xs[hit] = x + placed
    elif hit.any():
        xs[hit] = x
    return xs


def _leaders(ax, g: pd.DataFrame, xs: np.ndarray, x: float, top: float) -> None:
    hit = (g["OnProgram"] >= PART_SHARE).to_numpy()
    programs = sorted(set(g["Program"][hit]))
    for i, program in enumerate(programs):
        label_x = x + (i - (len(programs) - 1) / 2) * PROGRAM_STEP
        mine = hit & (g["Program"].to_numpy() == program)
        for px, py in zip(xs[mine], g["OnProgram"].to_numpy()[mine], strict=True):
            ax.plot([px, label_x], [py, top], color="0.45", lw=0.35, zorder=2)
        ax.text(
            label_x,
            top + 0.02,
            f"C{program}",
            ha="center",
            va="bottom",
            fontsize=LABEL_SIZE,
        )


def overlap(ax) -> None:
    _source(
        "comparator_column_program_overlap",
        "OnProgram",
        "Each comparator LR column not contained in cccRISE's span: its largest share "
        "of squared loading on one program's carrying pairs.",
    )
    left = [k for k in COLUMN_CLASSES if k != "Same as a cccRISE program"]

    def place(g: pd.DataFrame, x: float) -> np.ndarray:
        xs = _spread(g, x)
        fine_stac = (g["Method"] == "STAC").all() and (g["Labelling"] == "fine").all()
        _leaders(ax, g, xs, x, LEADER_TOP_HIGH if fine_stac else LEADER_TOP)
        return xs

    positions, spans = _strip(ax, "OnProgram", left, place)
    _threshold(ax, PART_SHARE)
    ax.set(ylim=(-0.04, 1.46), yticks=[0, 0.5, 1])
    _fit_axis(
        ax, positions, spans, "Carrying-pair share", "Overlap with one cccRISE program"
    )
    key_room(ax)
