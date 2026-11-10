"""The pseudobulk ceiling behind Figure 5A (`figureS_comparator_map` A and B):
per component and side, the share of its per-cell weight's variance that
(sample x cell type) means hold, the larger over coarse and fine cell types,
beside what the closest PB or STAC LR column of either labelling (greatest LR
|cosine|) holds of it, painted on cells as the cell's sample loading x its
cell-type loading (`run_comparator_scores`' ``label_ceiling``)."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS, load
from ._data import short

#: Pseudobulk's share of a per-cell weight's variance, Figure 5A's y and A/B's x.
R2_LABEL = "Pseudobulk variance\nexplained ($R^2$)"
MARKER_AREA = 8
#: Rows between the two cohorts' blocks.
BLOCK_GAP = 0.5


def best_rows(side: str) -> pd.DataFrame:
    """Per cohort and component on `side`: the larger ceiling over the
    labellings, and the closest column's row of either labelling."""
    frames = []
    for dataset, name in DATASET_LABELS.items():
        t = load("label_ceiling", dataset=dataset)
        t = t[t["Side"] == side]
        closest = t.loc[t.groupby("Component")["Cosine"].idxmax()].set_index(
            "Component"
        )
        frames.append(
            closest.assign(
                Ceiling=t.groupby("Component")["Ceiling"].max(), Dataset=name
            ).reset_index()
        )
    return pd.concat(frames, ignore_index=True)


def _source(t: pd.DataFrame, side: str) -> None:
    named = t.assign(
        Program="C" + t["Component"].astype(str),
        Name=[
            f"{short(m, c)} {lab[0].upper()}"
            for m, c, lab in zip(t["Method"], t["Column"], t["Labelling"], strict=True)
        ],
    )
    columns = {
        "Dataset": "Dataset",
        "Program": "Component",
        "Ceiling": "Pseudobulk ceiling R2",
        "Name": "Closest comparator column",
        "Cosine": "LR |cosine|",
        "Painted": "Closest comparator column R2",
    }
    common.source_data(
        f"pseudobulk_ceiling_{side}",
        named[list(columns)].rename(columns=columns),
        f"Per component, the share of its per-cell {side} weight's variance that "
        "(sample x cell type) means hold, the larger over coarse and fine cell types "
        "(Pseudobulk ceiling R2), and the squared correlation of that weight with the "
        "PB or STAC LR column of either labelling closest in LR content (greatest LR "
        "|cosine|), painted on cells as the cell's sample loading x its cell-type "
        "loading (Closest comparator column R2).",
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


def ceiling(ax, side: str) -> None:
    """`side`'s ceilings, one row per component, cohorts stacked."""
    t = best_rows(side)
    _source(t, side)
    centers, top = {}, 0.0
    for dataset, name in DATASET_LABELS.items():
        rows = t[t["Dataset"] == name].sort_values("Component")
        centers[dataset] = top + np.arange(len(rows)) + 0.5
        for y, (_, r) in zip(centers[dataset], rows.iterrows(), strict=True):
            ax.hlines(y, 0, r.Ceiling, color=style.COLORS["muted"], lw=2)
            ax.scatter(r.Ceiling, y, s=MARKER_AREA, color=style.COLORS["ink"], zorder=3)
            ax.scatter(
                r.Painted,
                y,
                s=MARKER_AREA,
                facecolor="white",
                edgecolor=style.COLORS[side],
                linewidths=0.8,
                zorder=4,
            )
        top += len(rows) + BLOCK_GAP
    ax.set(ylim=(top - BLOCK_GAP, 0), xlim=(0, 1), xticks=[0, 0.5, 1])
    ax.set_yticks(
        np.concatenate(list(centers.values())),
        [f"C{i + 1}" for y in centers.values() for i in range(len(y))],
        fontsize=style.FONT_SIZE_DENSE,
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(R2_LABEL)
    ax.set_title(side.title(), fontsize=style.FONT_SIZE_LABEL, fontweight="bold")
    _cohort_labels(ax, centers)


def key(ax) -> None:
    """The ceiling and the closest column, inside `ax`'s lower right."""
    ink = style.COLORS["ink"]
    ax.legend(
        handles=[
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
                label="Pseudobulk ceiling",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                mfc="white",
                mec=ink,
                ms=3,
                label="Closest comparator column",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        fontsize=style.FONT_SIZE_DENSE - 0.7,
        handlelength=1.2,
        handletextpad=0.3,
        borderaxespad=0,
    )
