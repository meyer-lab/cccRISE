"""
cccRISE against each comparator directly, on the permutations they share.

Plots `benchmark_anova_paired_<dataset>.csv`, written by
`uv run cccrise-benchmark --dataset <dataset>`. One panel per comparator rank
mode. Each row is an association; each comparator's marker sits at cccRISE's
best-component eta-squared minus that comparator's, over a gray bar spanning
the central 95% of the same difference across the shared label shuffles. The
bar is centred wherever the two methods' component counts put chance, not at
zero. A filled marker means the difference exceeds its paired null (one-sided
paired permutation p < 0.05). Rows follow `figureBenchmarkNull`: associations cccRISE
finds against its own null come first, shaded.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import getSetup, load_runner_csv, subplotLabel
from .figureBenchmarkNull import (
    ALPHA,
    HIGHLIGHT_COLOR,
    METHOD_STYLE,
    NULL_COLOR,
    NULL_CSV,
    RANK_MODE_TITLES,
    SELECTED_DATASET,
    _association_order,
)

PAIRED_CSV = f"benchmark_anova_paired_{SELECTED_DATASET}.csv"


def plot_paired_differences(
    df: pd.DataFrame,
    ax,
    order: list[str],
    found: set[str],
    title: str,
    show_labels: bool = True,
) -> None:
    """cccRISE minus each comparator, against the paired null."""
    comparators = [m for m in METHOD_STYLE if (df["Comparator"] == m).any()]
    offsets = np.linspace(-0.25, 0.25, len(comparators))
    y_of = {a: i for i, a in enumerate(order)}

    for association in found:
        ax.axhspan(
            y_of[association] - 0.5,
            y_of[association] + 0.5,
            color=HIGHLIGHT_COLOR,
            zorder=0,
        )
    ax.axvline(0, color="0.45", linewidth=0.8, zorder=1)

    for comparator, offset in zip(comparators, offsets):
        color, marker = METHOD_STYLE[comparator]
        rows = (
            df[df["Comparator"] == comparator].set_index("Association").reindex(order)
        )
        y = np.arange(len(order)) + offset
        ax.hlines(
            y,
            rows["NullDiffQ025"],
            rows["NullDiffQ975"],
            color=NULL_COLOR,
            linewidth=4,
            zorder=1,
        )
        ax.scatter(rows["NullDiffMean"], y, marker="|", s=40, color="0.55", zorder=2)
        significant = (rows["PairedP"] < ALPHA).to_numpy()
        ax.scatter(
            rows["Difference"][~significant],
            y[~significant],
            marker=marker,
            s=42,
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            zorder=3,
        )
        ax.scatter(
            rows["Difference"][significant],
            y[significant],
            marker=marker,
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        for diff, yy, p in zip(
            rows["Difference"][significant],
            y[significant],
            rows["PairedP"][significant],
        ):
            ax.annotate(
                f"p={p:.3f}",
                (diff, yy),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=6.5,
                color="0.25",
            )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order if show_labels else [])
    for label in ax.get_yticklabels():
        if label.get_text() in found:
            label.set_fontweight("bold")
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"cccRISE $-$ comparator, best-component $\eta^2$")
    ax.set_title(title, pad=14)


def makeFigure() -> Figure:
    df = load_runner_csv(PAIRED_CSV)
    order, found = _association_order(load_runner_csv(NULL_CSV))
    modes = [m for m in RANK_MODE_TITLES if (df["RankMode"] == m).any()]

    ax, f = getSetup((5.2 * len(modes) + 2.4, 6.2), (1, len(modes)))
    values = df[["Difference", "NullDiffQ025", "NullDiffQ975"]].to_numpy()
    span = float(np.nanmax(np.abs(values))) * 1.25
    for i, mode in enumerate(modes):
        panel = df[df["RankMode"] == mode]
        plot_paired_differences(
            panel,
            ax[i],
            order,
            found,
            title=RANK_MODE_TITLES[mode],
            show_labels=i == 0,
        )
        ax[i].set_xlim(-span, span)
        ranks = panel.drop_duplicates("Comparator").set_index("Comparator")[
            "ComparatorRank"
        ]
        ax[i].text(
            0.5,
            1.01,
            "ranks: "
            + ", ".join(f"{m} {ranks[m]}" for m in METHOD_STYLE if m in ranks),
            transform=ax[i].transAxes,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="0.35",
        )

    handles: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=6,
            label=f"vs. {method}",
        )
        for method, (color, marker) in METHOD_STYLE.items()
        if (df["Comparator"] == method).any()
    ]
    handles += [
        Line2D(
            [],
            [],
            color="0.3",
            marker="o",
            linestyle="none",
            markersize=6,
            label=f"filled: above paired null, p < {ALPHA}",
        ),
        Line2D(
            [],
            [],
            color="0.3",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            markersize=6,
            label="hollow: not significant",
        ),
        Line2D([], [], color=NULL_COLOR, linewidth=4, label="paired null: central 95%"),
        Patch(facecolor=HIGHLIGHT_COLOR, edgecolor="0.7", label="cccRISE association"),
    ]
    f.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    subplotLabel(ax)
    return f
