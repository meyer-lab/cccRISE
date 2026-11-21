"""Shared inputs of Figure 5 and its supplements: the comparator scores
(``uv run cccrise-comparator-scores``), on ALAD unless a panel names IBD, and
how each method and dataset is named and colored."""

import pandas as pd
from matplotlib.lines import Line2D

from ...runners.datasets import COMPARATOR_SCORES_DIR, scores_filename
from .. import common, style

DATASET = "alad"
HINT = "    uv run cccrise-comparator-scores --dataset <dataset>"
#: The cohorts a panel may draw together, in order.
DATASET_LABELS = {"alad": "ALAD", "ibd": "IBD"}
DATASET_COLORS = {"alad": style.CATEGORICAL[6], "ibd": style.CATEGORICAL[5]}
#: Each cell-type labelling's color: an indigo, light for coarse and deep for
#: fine, a hue no cohort, method, cell type or direction uses. Not drawn yet:
#: kept, by request, for marking labellings in later figures.
LABELLING_COLORS = {"coarse": "#a3a6e8", "fine": "#3b3fa8"}
LABELLING_NAMES = {"coarse": "Coarse", "fine": "Fine"}


def cohort_handles() -> list[Line2D]:
    """A legend handle per cohort: a dot in its color."""
    return [
        Line2D([], [], marker="o", ls="", ms=3, color=c, label=DATASET_LABELS[d])
        for d, c in DATASET_COLORS.items()
    ]


#: Comparators of the main figure, in column order.
COMPARATORS = ("PB", "TC2C", "STAC")
METHOD_COLORS = {
    "cccRISE": style.COLORS["emphasis"],
    "PB": style.CATEGORICAL[1],
    "TC2C": style.CATEGORICAL[2],
    "STAC": style.CATEGORICAL[3],
}


def load(
    what: str, labelling: str | None = None, dataset: str = DATASET
) -> pd.DataFrame:
    """One scores table (`run_comparator_scores`): every labelling's rows,
    or only `labelling`'s, without the ``Labelling`` column."""
    t = common.load_runner_output(
        scores_filename(what, dataset), COMPARATOR_SCORES_DIR, HINT
    )
    if labelling is None:
        return t
    rows = t[t["Labelling"] == labelling]
    return rows.drop(columns="Labelling").reset_index(drop=True)


#: A cohort bar's height over its axes (in)
COHORT_BAR_GAP = 2 * style.MM


def cohort_bar(ax, dataset: str, left: float, right: float) -> None:
    """A bar in `dataset`'s color `COHORT_BAR_GAP` over `ax`, from data x
    `left` to `right`, its name over it; drawn in `ax`, so its margins hold it."""
    height = ax.bbox.height / ax.figure.dpi
    bar = 1 + COHORT_BAR_GAP / height
    ax.plot(
        [left, right],
        [bar, bar],
        color=DATASET_COLORS[dataset],
        lw=2.5,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(
        (left + right) / 2,
        bar + style.MM / height,
        DATASET_LABELS[dataset],
        ha="center",
        va="bottom",
        fontsize=style.FONT_SIZE_DENSE,
        transform=ax.get_xaxis_transform(),
    )
