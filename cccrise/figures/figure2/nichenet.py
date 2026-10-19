"""NicheNet downstream-target validation of CP components, from
``uv run cccrise-nichenet``: do the ligands a component weights most have the
best-supported downstream targets in its receiver cells?

Per component, the weight-averaged target-score percentile of its ligands
(``nichenet.weighted_percentile``; diamond) against the same average with
every ligand re-scored on copies of the NicheNet prior with its ligand-target
links shuffled at random (``nichenet.shuffled_percentiles``; box: quartiles,
whiskers at the 2.5th and 97.5th percentiles of the shuffles). Above each
component: a star
where the one-sided permutation p, Benjamini-Hochberg adjusted across
components, is below `SIG_ALPHA` (``nichenet.weighted_percentile_tests``),
else n.s. The per-ligand scores and percentiles behind it are in the
NicheNet supplement (``figureS_nichenet``)."""

import functools

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.nichenet import weighted_percentile_tests
from .. import common, style

DATASET = "alad"
NULL_WHISKERS = (2.5, 97.5)
TEST_PAD = 0.04  # significance mark height above a component's data
BOX_FACE = "0.95"
BOX_STYLE = {
    "widths": 0.6,
    "showfliers": False,
    "patch_artist": True,
    "boxprops": {
        "facecolor": BOX_FACE,
        "edgecolor": style.COLORS["reference"],
        "lw": style.LINE_WIDTH_THIN,
    },
    "medianprops": {"color": style.COLORS["ink"], "lw": style.LINE_WIDTH},
    "whiskerprops": {"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
    "capprops": {"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
}
AXES = ("rect",)


@functools.cache
def load_scores() -> pd.DataFrame:
    """The runner's `component_ligand_scores` table."""
    return _load(DATASETS[DATASET].nichenet_scores_filename())


@functools.cache
def load_null() -> pd.DataFrame:
    """The runner's `shuffled_percentiles` table."""
    return _load(DATASETS[DATASET].nichenet_null_filename())


def _load(filename: str) -> pd.DataFrame:
    return common.load_runner_output(
        filename,
        VALIDATION_DIR,
        hint=f"    uv run cccrise-nichenet --dataset {DATASET}",
    )


def plot(axs):
    (ax,) = axs
    tests, null = weighted_percentile_tests(load_scores(), load_null())
    tests = tests.set_index("component")
    components = list(tests.index)
    positions = range(len(components))
    ax.boxplot(
        [null.loc[c] for c in components],
        positions=positions,
        whis=NULL_WHISKERS,
        zorder=1,
        **BOX_STYLE,
    )
    ax.scatter(
        positions,
        tests["weighted_pct"],
        marker="D",
        s=style.POINT_SIZE_FOREGROUND,
        color=style.COLORS["emphasis"],
        zorder=3,
    )
    for x, c in zip(positions, components):
        top = max(tests.loc[c, "weighted_pct"], null.loc[c].quantile(0.975))
        significant = tests.loc[c, "q"] < common.SIG_ALPHA
        ax.text(
            x,
            top + TEST_PAD,
            "*" if significant else "n.s.",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if significant else "normal",
            color=style.COLORS["ink"] if significant else style.COLORS["reference"],
        )
    common.reference_line(ax, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(positions, components)
    ax.set_xlim(-0.6, len(components) - 0.4)
    ax.set_xlabel("CP component")
    ax.set_ylabel("Weighted NicheNet\ntarget-score percentile")
    ax.legend(
        loc="lower center",
        ncols=2,
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                ls="",
                color=style.COLORS["emphasis"],
                markersize=style.MARKER_SIZE,
                label="Observed",
            ),
            Patch(
                facecolor=BOX_FACE,
                edgecolor=style.COLORS["reference"],
                lw=style.LINE_WIDTH_THIN,
                label="Shuffled NicheNet",
            ),
        ],
    )


def makeFigure():
    return common.standalone_fixed(__name__)
