"""Figure 4H: NICHES-positive cells in cccRISE's populations against what their
fine labels predict, ALAD and IBD (``uv run cccrise-niches``)."""

import pandas as pd
from matplotlib.lines import Line2D

from ...runners.datasets import NICHES_DIR, scores_filename
from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS
from ._recovery import panel_title

AXES = ("fig4_niches",)
#: The populations of Figure 4F and G, and where their labels sit.
LEADERS = [
    ("ibd", 3, "sender", "C3 S", (0.3, 0.93)),
    ("ibd", 4, "sender", "C4 S", (0.06, 0.84)),
    ("ibd", 3, "receiver", "C3 R", (0.6, 0.47)),
    ("ibd", 4, "receiver", "C4 R", (0.41, 0.22)),
    ("alad", 5, "receiver", "C5 R", (0.13, 0.45)),
    ("alad", 5, "sender", "C5 S", (0.1, 0.025)),
]
TEXT_SIZE = style.FONT_SIZE_DENSE - 1


def populations() -> pd.DataFrame:
    return pd.concat(
        common.load_runner_output(
            scores_filename("populations", dataset),
            NICHES_DIR,
            f"uv run cccrise-niches --dataset {dataset}",
        ).assign(Dataset=dataset)
        for dataset in DATASET_LABELS
    )


def _leaders(ax, t: pd.DataFrame) -> None:
    for dataset, component, side, text, at in LEADERS:
        r = t[
            (t["Dataset"] == dataset)
            & (t["Component"] == component)
            & (t["Side"] == side)
        ].iloc[0]
        ax.annotate(
            text,
            (r["Labels fine"], r["Observed"]),
            xytext=at,
            fontsize=TEXT_SIZE,
            fontweight="bold",
            va="center",
            zorder=5,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.35", "shrinkB": 1.5},
        )


def _key() -> list:
    def handle(label, **kwargs):
        return Line2D([], [], marker="o", ls="", ms=3, label=label, **kwargs)

    return [
        *[handle(DATASET_LABELS[d], color=c) for d, c in DATASET_COLORS.items()],
        handle("q < 0.05", color="0.4"),
        handle("q ≥ 0.05", mfc="white", mec="0.4"),
    ]


def plot(axs):
    ax = axs[0]
    t = populations()
    common.source_data(
        "niches_positive_fraction_by_population",
        t[["Dataset", "Component", "Side", "Observed", "Labels fine", "Q direction"]],
        "Share of each population's cells with a non-zero NICHES program score "
        "(Observed), the share its fine labels predict, and q against random directions.",
    )
    for dataset, g in t.groupby("Dataset"):
        hit = g["Q direction"] < common.SIG_ALPHA
        color = DATASET_COLORS[dataset]
        ax.scatter(
            g["Labels fine"],
            g["Observed"],
            s=12,
            lw=0.7,
            zorder=3,
            facecolor=[color if h else "white" for h in hit],
            edgecolor=color,
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color=style.COLORS["reference"],
        ls=style.LINE_STYLE_REFERENCE,
        lw=0.6,
        zorder=1,
    )
    _leaders(ax, t)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xticks=[0, 0.5, 1],
        yticks=[0, 0.5, 1],
        xlabel="Fine labels",
        ylabel="CCC-RISE",
    )
    panel_title(ax, "NICHES concordance")
    ax.legend(handles=_key(), loc="lower right", fontsize=TEXT_SIZE, labelspacing=0.08)


def makeFigure():
    return common.standalone_fixed(__name__)
