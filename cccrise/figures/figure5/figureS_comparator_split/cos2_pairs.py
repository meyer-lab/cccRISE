"""Figure EV5A: each CCC-RISE component's best single comparator LR column
(its largest cos^2 with one LR column of any PB, TC2C or STAC fit) against
how many LR pairs carry it (`covariation.carrying_pairs`; their Spearman p
over the legend, rho in the source data), with, dashed, the axes' anti-diagonal: a perfect
negative ordering (rho = -1). Components sharing a pair count and close in
cos^2 are dodged apart on the log axis (`dodged`)."""

import numpy as np
import pandas as pd
from matplotlib import patheffects
from scipy.stats import spearmanr

from .... import covariation
from ... import common, style
from .._common import DATASET_COLORS, DATASET_LABELS, cohort_handles
from .._programs import best_columns, carrying_pairs
from .._recovery import COS2_LABEL, LABEL_SIZE

AXES = ("split_cos2_pairs",)
#: Components sharing an LR pair count within this cos^2 are dodged apart, a
#: factor of exp(`DODGE`) on the log axis.
TIE = 0.06
DODGE = 0.09
XLIM = (0.75, 18)
#: Each label's text, offset (points) from its components' mean position and
#: alignment, by (cohort, components); a tie gets one label.
LABELS = {
    ("ibd", (2, 3, 4)): ("C2–4", (6, 1), "left"),
    ("ibd", (6,)): ("C6", (4, -2), "left"),
    ("alad", (1,)): ("C1", (4, 0), "left"),
    ("alad", (6,)): ("C6", (0, -7), "center"),
    ("alad", (2,)): ("C2", (4, 0), "left"),
    ("alad", (3,)): ("C3", (-5, 0), "right"),
    ("alad", (5,)): ("C5", (5, 0), "left"),
    ("alad", (4,)): ("C4", (5, 0), "left"),
    ("ibd", (1,)): ("C1", (-5, 0), "right"),
    ("ibd", (5,)): ("C5", (5, 0), "left"),
}


def component_rows() -> pd.DataFrame:
    """Per cohort and component: its largest cos^2 with one LR column of any
    fit, that fit, and its number of carrying LR pairs."""
    frames = []
    for dataset in DATASET_LABELS:
        best = best_columns(dataset)
        best = best.loc[best.groupby("Program")["Cos2"].idxmax()]
        pairs = carrying_pairs(dataset)
        frames.append(
            best.assign(
                Dataset=dataset,
                Fit=best["Method"] + " " + best["Labelling"],
                **{"Carrying pairs": [len(pairs[p]) for p in best["Program"]]},
            )
        )
    return pd.concat(frames, ignore_index=True)


def _runs(t: pd.DataFrame) -> list[list[int]]:
    """Index runs of components with the same pair count, chained while
    consecutive cos^2 lie within `TIE`."""
    runs = []
    for _, g in t.groupby("Carrying pairs"):
        order = g.sort_values("Cos2").index.to_list()
        run = [order[0]]
        for i in order[1:]:
            if t.loc[i, "Cos2"] - t.loc[run[-1], "Cos2"] < TIE:
                run.append(i)
            else:
                runs.append(run)
                run = [i]
        runs.append(run)
    return runs


def dodged(t: pd.DataFrame) -> np.ndarray:
    """x = carrying pairs, each run of ties (`_runs`) spread evenly on the log
    axis."""
    x = t["Carrying pairs"].to_numpy(float).copy()
    for run in _runs(t):
        at = t.index.get_indexer(run)
        x[at] *= np.exp((np.arange(len(run)) - (len(run) - 1) / 2) * DODGE)
    return x


def _labels(ax, t: pd.DataFrame) -> None:
    for (dataset, programs), (text, offset, ha) in LABELS.items():
        rows = t[(t["Dataset"] == dataset) & t["Program"].isin(programs)]
        ax.annotate(
            text,
            (rows["X"].mean(), rows["Cos2"].mean()),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=LABEL_SIZE,
            color=DATASET_COLORS[dataset],
            zorder=4,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
        )


def plot(axs):
    (ax,) = axs
    t = component_rows()
    t["X"] = dodged(t)
    result = spearmanr(t["Carrying pairs"], t["Cos2"])
    common.source_data(
        "component_cos2_and_carrying_pairs",
        t[["Dataset", "Program", "Cos2", "Fit", "Carrying pairs"]].rename(
            columns={"Program": "Component", "Cos2": "Best-match cos2"}
        ),
        "Per component: its largest cos^2 with one LR column of any comparator "
        "cell-type fit (Fit names it), and its number of carrying LR pairs (the "
        "fewest top pairs carrying 90% of its squared LR loading, at most "
        f"{covariation.CARRYING_CAP}). Spearman rho over all {len(t)} components: "
        f"{result.statistic:.2f} (p {result.pvalue:.2g}); the dashed line is the "
        "axes' anti-diagonal, a perfect negative ordering (rho = -1).",
    )
    ax.plot(
        [0, 1],
        [1, 0],
        transform=ax.transAxes,
        color=style.COLORS["reference"],
        ls=style.LINE_STYLE_REFERENCE,
        lw=0.6,
        zorder=1,
    )
    for _, r in t.iterrows():
        ax.scatter(
            r["X"],
            r["Cos2"],
            s=14,
            color=DATASET_COLORS[r["Dataset"]],
            ec="white",
            lw=0.3,
            zorder=3,
        )
    _labels(ax, t)
    ax.set(
        xscale="log",
        xlim=XLIM,
        ylim=(0, 1.05),
        yticks=[0, 0.5, 1],
        xlabel="Number of top LR pairs",
        ylabel=COS2_LABEL,
    )
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.minorticks_off()
    ax.legend(
        handles=cohort_handles(),
        loc="lower left",
        fontsize=LABEL_SIZE,
        handletextpad=0.2,
        title=common.p_text(result.pvalue, "Spearman p"),
        title_fontsize=LABEL_SIZE,
        alignment="left",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
