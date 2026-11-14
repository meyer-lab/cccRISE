"""NicheNet downstream-target validation of CP components, from
``uv run cccrise-nichenet``: do a component's top receiver cells upregulate
the downstream targets of the ligands it weights?

Per component, every ligand's NicheNet target score on the DE of the
component's top receivers against the bottom receivers
(``nichenet.component_ligand_scores``; box, with the ligands as points colored
and sized by their weight in the component, opaque for the top ligands), beside
the same scores with the top/bottom labels shuffled among those cells
(``nichenet.shuffled_cell_scores``; grey box, pooled over the shuffles). Above
each component: a star where the ligand-weighted mean score exceeds its
cell-shuffle null (one-sided permutation p, Benjamini-Hochberg adjusted across
components, below `SIG_ALPHA`; ``nichenet.weighted_score_tests``), else n.s."""

import functools

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.nichenet import weighted_score_tests
from .. import common, style

DATASET = "alad"
TEST_PAD = 0.04  # significance mark height above a component's data
BOX_FACE = "0.95"
NULL_FACE = style.COLORS["muted"]
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
BOX_WIDTH = 0.38  # observed and shuffled boxes side by side
WEIGHT_FLOOR = 1e-3
JITTER = 0.22  # of a box's width-0.6 slot; scaled to the box drawn
POINT_SIZE_MAX = 8.4
BACKGROUND_ALPHA = 0.5  # non-top ligands, so the boxes show through
AXES = ("rect",)


@functools.cache
def load_scores(dataset: str = DATASET) -> pd.DataFrame:
    """The runner's `component_ligand_scores` table."""
    return _load(DATASETS[dataset].nichenet_scores_filename(), dataset)


@functools.cache
def load_cell_null(dataset: str = DATASET) -> pd.DataFrame:
    """The runner's `shuffled_cell_scores` table."""
    return _load(DATASETS[dataset].nichenet_cell_null_filename(), dataset)


def _load(filename: str, dataset: str) -> pd.DataFrame:
    return common.load_runner_output(
        filename,
        VALIDATION_DIR,
        hint=f"    uv run cccrise-nichenet --dataset {dataset}",
    )


def plot(axs):
    draw(axs, DATASET)


def draw(axs, dataset: str) -> None:
    """`dataset`'s per-ligand target scores per CP component against their
    cell-shuffle null."""
    (ax,) = axs
    scores = load_scores(dataset).dropna(subset=["auroc"])
    cell_null = load_cell_null(dataset).dropna(subset=["auroc"])
    tests, null = weighted_score_tests(load_scores(dataset), load_cell_null(dataset))
    _source_data(tests, null, scores)
    components = sorted(scores["component"].unique())
    x = np.arange(len(components))
    rng = np.random.default_rng(0)
    for i, c in enumerate(components):
        points = ligand_points(
            ax, scores[scores["component"] == c], x[i] - BOX_WIDTH / 2, rng, BOX_WIDTH
        )
    for offset, table, face in (
        (-BOX_WIDTH / 2, scores, BOX_FACE),
        (BOX_WIDTH / 2, cell_null, NULL_FACE),
    ):
        boxes = ax.boxplot(
            [table.loc[table["component"] == c, "auroc"] for c in components],
            positions=x + offset,
            zorder=1,
            **{**BOX_STYLE, "widths": BOX_WIDTH},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(face)
    # One row of marks over every component's points
    top = [scores["auroc"].max()] * len(components)
    q = tests.set_index("component")["q"].loc[components]
    significance_marks(ax, x, top, q, values=True)
    common.reference_line(ax, 0.5)
    ax.set_xticks(x, components)
    ax.set_xlim(-0.6, len(components) - 0.4)
    ax.set_xlabel("CPD component")
    ax.set_ylabel("NicheNet target score")
    common.colorbar(ax, points, "Relative ligand weight")
    common.legend(
        ax,
        "above",
        ncols=2,
        handles=[
            Patch(facecolor=f, edgecolor=style.COLORS["reference"], label=label)
            for label, f in (("Observed", BOX_FACE), ("Shuffled cells", NULL_FACE))
        ],
    )


def _source_data(tests: pd.DataFrame, null: pd.Series, scores: pd.DataFrame) -> None:
    common.source_data(
        "nichenet_target_score_per_ligand",
        scores[
            ["component", "ligand", "auroc", "weight", "rel_weight", "is_top"]
        ].rename(
            columns={
                "component": "CPD component",
                "ligand": "Ligand",
                "auroc": "NicheNet target score (AUROC)",
                "weight": "Ligand weight",
                "rel_weight": "Relative ligand weight",
                "is_top": "Top ligand",
            }
        ),
        "One row per ligand NicheNet scores and CPD component: the AUROC of the "
        "component's top vs. bottom receiver ranking for the ligand's "
        "downstream targets; the ligand's weight in the component (and relative "
        "to the component's largest, plotted floored at 1e-3); whether it is one "
        "of the component's top ligands.",
    )
    common.source_data(
        "nichenet_weighted_score_tests",
        tests,
        "Per CPD component: the ligand-weighted mean NicheNet target score "
        "(weighted_auroc), its mean over the cell-shuffled rankings (null_mean), "
        "and the one-sided permutation p, Benjamini-Hochberg adjusted across "
        "components (q).",
    )
    common.source_data(
        "nichenet_cell_shuffle_weighted_scores",
        null.rename("weighted_auroc"),
        "The ligand-weighted mean target score per CPD component on each ranking "
        "with the top/bottom receiver labels shuffled (replicate): the test's "
        "null. The grey boxes pool the per-ligand scores of these rankings.",
    )


def ligand_points(ax, component_scores, x, rng, width: float = 0.6):
    """Weight-colored and -sized points for one component's ligands at `x`,
    jittered within a box `width` wide, the non-top ones translucent; returns
    the (opaque) top ligands' points."""
    c = component_scores.assign(
        w=np.clip(component_scores["rel_weight"], WEIGHT_FLOOR, 1)
    ).sort_values("w")
    jitter = JITTER * width / 0.6
    c["x"] = x + rng.uniform(-jitter, jitter, len(c))
    for top, alpha in ((False, BACKGROUND_ALPHA), (True, 1.0)):
        subset = c[c["is_top"] == top]
        points = ax.scatter(
            subset["x"],
            subset["auroc"],
            c=subset["w"],
            cmap=style.CMAPS["weight"],
            norm=LogNorm(vmin=WEIGHT_FLOOR, vmax=1),
            s=style.POINT_SIZE_BACKGROUND / 2 + POINT_SIZE_MAX * subset["w"],
            linewidths=0,
            alpha=alpha,
            zorder=3 + top,
        )
    return points


def significance_marks(ax, positions, tops, q, values: bool = False) -> None:
    """A star above each of `positions` (at its data's top, `tops`) where `q`
    is below `SIG_ALPHA`, else n.s.; with `values`, `q` itself, greyed where
    not significant."""
    for x, top, value in zip(positions, tops, q, strict=True):
        significant = value < common.SIG_ALPHA
        if values:
            label = f"{value:#.2g}"
        else:
            label = "*" if significant else "n.s."
        ax.text(
            x,
            top + TEST_PAD,
            label,
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if significant and not values else "normal",
            color=style.COLORS["ink"] if significant else style.COLORS["reference"],
        )


def makeFigure():
    return common.standalone_fixed(__name__)
