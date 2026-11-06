"""Figure 4I and J: the label ceiling, the share of each cccRISE program's
per-cell weight that (sample x cell type) means hold, on ALAD and IBD at both
labellings, beside what the closest comparator column holds of it, painted
on cells as sample loading x type loading (`run_comparator_scores`'s
``label_ceiling``)."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS, load
from ._data import LABELLINGS, short

MARKERS = {"coarse": "s", "fine": "o"}
#: Each labelling's offset from its program's row, in rows.
OFFSETS = {"coarse": -0.2, "fine": 0.2}
#: Rows between the two cohorts' blocks.
BLOCK_GAP = 0.5
MARKER_AREA = 8


def closest_columns(side: str) -> pd.DataFrame:
    """Per cohort, labelling and program on `side`, the ceiling and the
    `label_ceiling` row of the comparator column closest in LR content."""
    frames = []
    for dataset, name in DATASET_LABELS.items():
        t = load("label_ceiling", dataset=dataset)
        t = t[t["Side"] == side]
        keys = ["Labelling", "Component"]
        frames.append(t.loc[t.groupby(keys)["Cosine"].idxmax()].assign(Dataset=name))
    return pd.concat(frames, ignore_index=True)


def _source(t: pd.DataFrame, side: str) -> None:
    columns = {
        "Dataset": "Dataset",
        "Program": "Program",
        "Labelling": "Labelling",
        "Ceiling": "Label ceiling R2",
        "Name": "Closest column",
        "Cosine": "LR |cosine|",
        "Painted": "Closest column R2",
    }
    named = t.assign(
        Program="C" + t["Component"].astype(str),
        Name=[short(m, c) for m, c in zip(t["Method"], t["Column"], strict=True)],
    )
    common.source_data(
        f"label_ceiling_{side}",
        named[list(columns)].rename(columns=columns),
        f"Per program, the share of its per-cell {side} weight's variance that "
        "(sample x cell type) means hold, the most any method pooling cells by "
        "sample and annotation could represent (Label ceiling R2), and the "
        "squared correlation of that weight with the closest PB or STAC LR "
        "column (greatest LR |cosine|), painted on cells as the cell's sample "
        "loading x its type loading (Closest column R2).",
    )


def _cohort_labels(ax, centers: dict) -> None:
    """Each cohort's name beside a bar in its color, right of `ax`."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    width = ax.bbox.width / ax.figure.dpi
    bar, text = 1 + 1.2 * style.MM / width, 1 + 2.4 * style.MM / width
    for dataset, y in centers.items():
        ax.plot(
            [bar, bar],
            [y[0] - 0.4, y[-1] + 0.4],
            color=DATASET_COLORS[dataset],
            lw=2.5,
            transform=trans,
            clip_on=False,
            solid_capstyle="butt",
        )
        ax.text(
            text,
            y.mean(),
            DATASET_LABELS[dataset],
            rotation=-90,
            ha="left",
            va="center",
            fontsize=style.FONT_SIZE_LABEL,
            transform=trans,
        )


def _draw_program(ax, rows: pd.DataFrame, y: float, side: str) -> None:
    for labelling, top, held in zip(
        rows["Labelling"], rows["Ceiling"], rows["Painted"], strict=True
    ):
        at = y + OFFSETS[labelling]
        marker = MARKERS[labelling]
        ax.hlines(at, 0, top, color=style.COLORS["muted"], lw=2)
        ax.scatter(
            top, at, marker=marker, s=MARKER_AREA, color=style.COLORS["ink"], zorder=3
        )
        ax.scatter(
            held,
            at,
            marker=marker,
            s=MARKER_AREA,
            facecolor="white",
            edgecolor=style.COLORS[side],
            linewidths=0.8,
            zorder=4,
        )


def ceiling(ax, side: str) -> dict:
    """`side`'s ceilings, one row per program, cohorts stacked; the rows'
    centers by cohort."""
    t = closest_columns(side)
    _source(t, side)
    centers, top = {}, 0.0
    for dataset, name in DATASET_LABELS.items():
        rows = t[t["Dataset"] == name]
        programs = sorted(rows["Component"].unique())
        centers[dataset] = top + np.arange(len(programs)) + 0.5
        for y, k in zip(centers[dataset], programs, strict=True):
            _draw_program(ax, rows[rows["Component"] == k], y, side)
        top += len(programs) + BLOCK_GAP
    ax.set_ylim(top - BLOCK_GAP, 0)
    ax.set_xlim(0, 1)
    ax.set_yticks(
        np.concatenate(list(centers.values())),
        [f"C{i + 1}" for y in centers.values() for i in range(len(y))],
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Per-cell weight $R^2$")
    ax.set_title(side.title(), fontsize=style.FONT_SIZE_LABEL, fontweight="bold")
    _cohort_labels(ax, centers)
    return centers


def key(ax, inside: bool = False) -> None:
    """The panels' key, right of `ax`'s cohort labels, or `inside` its lower
    right corner."""
    ink = style.COLORS["ink"]
    handles = [
        *[
            Line2D(
                [],
                [],
                ls="",
                marker=MARKERS[lab],
                mfc="none",
                mec=ink,
                ms=3,
                label=f"{lab.title()} labels",
            )
            for lab in LABELLINGS
        ],
        Line2D(
            [0, 1],
            [0, 0],
            color=style.COLORS["muted"],
            lw=2,
            marker="o",
            mfc=ink,
            mec=ink,
            ms=3,
            markevery=[1],
            label="Label ceiling",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="white",
            mec=ink,
            ms=3,
            label="Closest column" if inside else "Closest PB or\nSTAC column",
        ),
    ]
    if inside:
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1.03, -0.02),
            fontsize=style.FONT_SIZE_DENSE - 1.4,
            labelspacing=0.2,
            handlelength=1.2,
            handletextpad=0.3,
            borderaxespad=0,
        )
    else:
        common.legend(ax, "right", handles=handles, fontsize=style.FONT_SIZE_DENSE)
