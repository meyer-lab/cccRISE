"""Figure 5A: each CCC-RISE component's best single comparator LR column
against its pseudobulk ceiling, both on the best reading across cell-type
fits (coarse or fine). Components whose max cos^2 tie and overlap on y
are spread apart on x (`jittered`); source data keep the
true cos^2."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .. import common
from ._ceiling import R2_LABEL
from ._common import DATASET_COLORS, DATASET_LABELS, load
from ._programs import best_columns
from ._recovery import COS2_LABEL, LABEL_SIZE, panel_title

AXES = ("fig5_map",)
TITLE = "Pseudobulk Representation of CCC-RISE Components"
#: Components labelled left of their point.
LEFT_LABELS = {("ibd", 1), ("ibd", 3), ("ibd", 6), ("alad", 2), ("alad", 5)}
#: Max cos^2 closer than this count as tied.
TIE = 0.012
#: Spacing between tied components on x.
STEP = 0.022
#: Sender-receiver spans closer than this on y count as overlapping.
PAD = 0.05


def _overlap_groups(cos2, low, high) -> list[list[int]]:
    """Components that would overlap, chained into groups, each by cos^2: two
    overlap when their max cos^2 lie within `TIE` and their sender-receiver
    spans [`low`, `high`] within `PAD`."""
    n = len(cos2)
    group = list(range(n))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
    for i, j in pairs:
        if abs(cos2[i] - cos2[j]) < TIE and max(low[i], low[j]) - PAD <= min(
            high[i], high[j]
        ):
            group[root(j)] = root(i)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(root(i), []).append(i)
    return [sorted(m, key=lambda i: cos2[i]) for m in members.values()]


def jittered(cos2, low, high) -> np.ndarray:
    """`cos2` with each group of overlapping components (`_overlap_groups`)
    spread `STEP` apart about its mean, kept at or below 1."""
    cos2, low, high = (np.asarray(v, float) for v in (cos2, low, high))
    x = cos2.copy()
    for run in _overlap_groups(cos2, low, high):
        x[run] = cos2[run].mean() + (np.arange(len(run)) - (len(run) - 1) / 2) * STEP
        x[run] -= max(x[run].max() - 1, 0)
    return x


def best_of_rows() -> pd.DataFrame:
    """Per component, its largest cos^2 with any comparator LR column of any
    cell-type fit, that column's fit, and the largest R2 of each side's
    per-cell weight on sample x cell type over both labellings."""
    frames = []
    for dataset in DATASET_LABELS:
        best = best_columns(dataset)
        best = best.loc[best.groupby("Program")["Cos2"].idxmax()].set_index("Program")
        ceiling = load("label_ceiling", dataset=dataset)
        r2 = ceiling.groupby(["Component", "Side", "Labelling"])["Ceiling"].first()
        r2 = r2.groupby(["Component", "Side"]).max().unstack()
        frames.append(
            r2.assign(
                Dataset=dataset,
                Cos2=best["Cos2"],
                Fit=best["Method"] + " " + best["Labelling"],
            )
            .rename_axis("Program")
            .reset_index()
        )
    t = pd.concat(frames, ignore_index=True)
    low = t[["sender", "receiver"]].min(axis=1)
    high = t[["sender", "receiver"]].max(axis=1)
    return t.assign(X=jittered(t["Cos2"], low, high))


def _key() -> list:
    def handle(label, color, marker="o"):
        return Line2D([], [], marker=marker, ls="", color=color, ms=3, label=label)

    return [
        *[handle(DATASET_LABELS[d], c) for d, c in DATASET_COLORS.items()],
        handle("Sender", "0.4"),
        handle("Receiver", "0.4", "s"),
    ]


def plot(axs):
    ax = axs[0]
    t = best_of_rows()
    common.source_data(
        "program_cos2_and_pseudobulk_ceiling",
        t.rename(
            columns={"sender": "Sender R2", "receiver": "Receiver R2", "X": "Plotted x"}
        ),
        "Per component: its largest cos^2 with one LR column of any comparator "
        "cell-type fit (Fit names it), and the largest R2 over coarse and fine "
        "cell types of its per-cell sender and receiver weights on sample x cell "
        "type means. Plotted x is cos^2 with tied components spread apart for display.",
    )
    for _, r in t.iterrows():
        color = DATASET_COLORS[r.Dataset]
        ax.plot([r.X] * 2, [r.sender, r.receiver], color=color, lw=0.8, zorder=1)
        ax.scatter(r.X, r.sender, s=13, color=color, ec="white", lw=0.3, zorder=3)
        ax.scatter(
            r.X, r.receiver, s=11, marker="s", color=color, ec="white", lw=0.3, zorder=3
        )
        left = (r.Dataset, r.Program) in LEFT_LABELS
        ax.text(
            r.X + (-0.02 if left else 0.02),
            (r.sender + r.receiver) / 2,
            f"C{r.Program}",
            ha="right" if left else "left",
            va="center",
            fontsize=LABEL_SIZE,
        )
    ax.set(
        xlim=(0, 1.1),
        ylim=(0, 1),
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
        xlabel=COS2_LABEL,
        ylabel=R2_LABEL,
    )
    panel_title(ax, TITLE)
    ax.legend(handles=_key(), loc="upper left", fontsize=LABEL_SIZE)


def makeFigure():
    return common.standalone_fixed(__name__)
