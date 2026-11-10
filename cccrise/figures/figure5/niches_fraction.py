"""Figure 5H: each CCC-RISE population's fraction NICHES-positive against what
cell type predicts, each population cell at the NICHES-positive rate of its
fine type in its own sample, ALAD and IBD on one axes (``uv run
cccrise-niches``; `comparators.niches_programs.population_table`), with a
paired (Wilcoxon signed-rank, two-sided) test over every population of both."""

import pandas as pd
from scipy.stats import wilcoxon

from ...runners.datasets import NICHES_DIR, scores_filename
from .. import common, style
from ._common import DATASET_COLORS, DATASET_LABELS, cohort_handles
from ._recovery import panel_title

AXES = ("fig4_niches",)
#: The populations of Figure 5F and G, and where their labels sit.
LEADERS = [
    ("ibd", 3, "sender", "C3 S", (0.3, 0.93)),
    ("ibd", 4, "sender", "C4 S", (0.06, 0.84)),
    ("ibd", 3, "receiver", "C3 R", (0.6, 0.47)),
    ("ibd", 4, "receiver", "C4 R", (0.41, 0.22)),
    ("alad", 5, "receiver", "C5 R", (0.13, 0.45)),
    ("alad", 5, "sender", "C5 S", (0.1, 0.025)),
]
TEXT_SIZE = style.FONT_SIZE_DENSE - 1
PREDICTED = "Cell type predicted"


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
            (r[PREDICTED], r["Observed"]),
            xytext=at,
            fontsize=TEXT_SIZE,
            fontweight="bold",
            va="center",
            zorder=5,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.35", "shrinkB": 1.5},
        )


def plot(axs):
    ax = axs[0]
    t = populations()
    p = wilcoxon(t["Observed"], t[PREDICTED]).pvalue
    common.source_data(
        "niches_positive_fraction_by_population",
        t[["Dataset", "Component", "Side", "Cells", "Samples", "Observed", PREDICTED]]
        .rename(columns={"Observed": "Fraction NICHES-positive"})
        .assign(**{"Wilcoxon signed-rank p (all populations)": p}),
        "Each population's share of cells with a non-zero NICHES program score, and "
        "what cell type predicts: each population cell at the NICHES-positive rate of "
        "its fine cell type in its own sample; and the two-sided Wilcoxon signed-rank "
        f"p of the two, paired over all {len(t)} populations of both cohorts.",
    )
    for dataset, g in t.groupby("Dataset"):
        color = DATASET_COLORS[str(dataset)]
        ax.scatter(g[PREDICTED], g["Observed"], s=12, lw=0.7, zorder=3, color=color)
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
        xlabel=PREDICTED,
        ylabel="CCC-RISE",
    )
    panel_title(ax, "Fraction NICHES-positive")
    ax.text(
        0.97,
        0.13,
        f"Paired Wilcoxon\n{common.p_text(p)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=TEXT_SIZE,
    )
    ax.legend(
        handles=cohort_handles(),
        loc="lower right",
        fontsize=TEXT_SIZE,
        labelspacing=0.08,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
