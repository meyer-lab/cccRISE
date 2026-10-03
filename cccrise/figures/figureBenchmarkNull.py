"""
cccRISE ANOVA benchmark against each method's own permutation null.

Plots `benchmark_anova_null_<dataset>.csv`, written by
`uv run cccrise-benchmark --dataset <dataset>`. One panel per comparator rank
mode. Each row is an association; each method's marker sits at its best
component's eta-squared, over a gray bar spanning its own null (mean to 95th
percentile of the maximum eta-squared across its components, under the same
shuffled labels for every method). A filled marker has a permutation p < 0.05,
family-wise across that method's components. Associations cccRISE finds are
moved to the top and shaded.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..runners.datasets import DATA_DIR
from .common import getSetup, subplotLabel

SELECTED_DATASET = "alad"
NULL_CSV = f"benchmark_anova_null_{SELECTED_DATASET}.csv"
ALPHA = 0.05

# Categorical slots in fixed order, validated all-pairs for CVD separation
METHOD_STYLE = {
    "cccRISE": ("#2a78d6", "o"),
    "Pseudobulk CPD": ("#eb6834", "s"),
    "Tensor-cell2cell": ("#1baf7a", "D"),
    "STACCato": ("#4a3aa7", "^"),
}
RANK_MODE_TITLES = {"own": "Own rank selection", "shared": "Shared rank"}
NULL_COLOR = "0.82"
HIGHLIGHT_COLOR = "#e3ecf9"


def load_runner_csv(filename: str) -> pd.DataFrame:
    """Read a CSV the benchmark runner (`cccrise-benchmark`) wrote to the data
    directory; a missing file is an error naming the command to run."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Produce it with\n    uv run cccrise-benchmark --dataset alad"
        )
    return pd.read_csv(path)


def _association_order(df: pd.DataFrame) -> tuple[list[str], set[str]]:
    """Associations cccRISE finds first, then the rest in their original order."""
    order = list(dict.fromkeys(df["Association"]))
    cccrise = df[df["Method"] == "cccRISE"].set_index("Association")["PermP"]
    found = {a for a in order if cccrise.get(a, np.nan) < ALPHA}
    return [a for a in order if a in found] + [
        a for a in order if a not in found
    ], found


def plot_null_comparison(
    df: pd.DataFrame,
    ax,
    order: list[str],
    found: set[str],
    title: str,
    show_labels: bool = True,
) -> None:
    """Best-component eta-squared per method against its own null."""
    methods = [m for m in METHOD_STYLE if (df["Method"] == m).any()]
    offsets = np.linspace(-0.3, 0.3, len(methods))
    y_of = {a: i for i, a in enumerate(order)}

    for association in found:
        ax.axhspan(
            y_of[association] - 0.5,
            y_of[association] + 0.5,
            color=HIGHLIGHT_COLOR,
            zorder=0,
        )

    for method, offset in zip(methods, offsets):
        color, marker = METHOD_STYLE[method]
        rows = df[df["Method"] == method].set_index("Association").reindex(order)
        y = np.arange(len(order)) + offset
        ax.hlines(
            y,
            rows["NullMean"],
            rows["NullQ95"],
            color=NULL_COLOR,
            linewidth=4,
            zorder=1,
        )
        significant = (rows["PermP"] < ALPHA).to_numpy()
        ax.scatter(
            rows["MaxEtaSquared"][~significant],
            y[~significant],
            marker=marker,
            s=42,
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            zorder=3,
        )
        ax.scatter(
            rows["MaxEtaSquared"][significant],
            y[significant],
            marker=marker,
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        for eta, yy, p in zip(
            rows["MaxEtaSquared"][significant],
            y[significant],
            rows["PermP"][significant],
        ):
            ax.annotate(
                f"p={p:.3f}",
                (eta, yy),
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
    ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"Best-component effect size ($\eta^2$)")
    ax.set_title(title)


def _rank_note(df: pd.DataFrame) -> str:
    ranks = df.drop_duplicates("Method").set_index("Method")["Rank"]
    return ", ".join(f"{m} {ranks[m]}" for m in METHOD_STYLE if m in ranks)


def makeFigure() -> Figure:
    df = load_runner_csv(NULL_CSV)
    cccrise = df[df["Method"] == "cccRISE"]
    modes = [m for m in RANK_MODE_TITLES if (df["RankMode"] == m).any()]
    order, found = _association_order(df)

    ax, f = getSetup((5.2 * len(modes) + 2.4, 6.2), (1, len(modes)))
    x_max = float(np.nanmax(df[["MaxEtaSquared", "NullQ95"]].to_numpy())) * 1.18
    for i, mode in enumerate(modes):
        panel = pd.concat([cccrise, df[df["RankMode"] == mode]])
        plot_null_comparison(
            panel,
            ax[i],
            order,
            found,
            title=RANK_MODE_TITLES[mode],
            show_labels=i == 0,
        )
        ax[i].set_xlim(0, x_max)
        ax[i].text(
            0.5,
            1.01,
            f"ranks: {_rank_note(panel)}",
            transform=ax[i].transAxes,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="0.35",
        )
        ax[i].set_title(RANK_MODE_TITLES[mode], pad=14)

    handles: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=6,
            label=method,
        )
        for method, (color, marker) in METHOD_STYLE.items()
        if (df["Method"] == method).any()
    ]
    handles += [
        Line2D(
            [],
            [],
            color="0.3",
            marker="o",
            linestyle="none",
            markersize=6,
            label=f"filled: permutation p < {ALPHA}",
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
        Line2D(
            [], [], color=NULL_COLOR, linewidth=4, label="own null: mean to 95th pct."
        ),
        Patch(facecolor=HIGHLIGHT_COLOR, edgecolor="0.7", label="cccRISE association"),
    ]
    f.legend(
        handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7
    )
    subplotLabel(ax)
    return f
