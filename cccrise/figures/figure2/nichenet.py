"""NicheNet downstream-target validation of CP components, from
``uv run cccrise-nichenet``: each component's top ligands against every other
ligand the model sees.

One box per component: the NicheNet-style AUROC
(``nichenet.component_ligand_scores``) of every scored ligand on the
component's receiver ranking, restricted to genes expressed in the receiver
groups. Points are colored and sized by the ligand's weight in the component,
opaque for the top ligands (``nichenet.top_pair_ligands``) and translucent for
the rest; the top LR pairs are labeled on `cp_factor_matrices`. Above each box: a
star where a two-sided Mann-Whitney test of the top ligands against all the
others has Benjamini-Hochberg q (across components) below `SIG_ALPHA`
(``nichenet.top_ligand_tests``), else n.s."""

import numpy as np
from matplotlib.colors import LogNorm

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.nichenet import top_ligand_tests
from .. import common, style

DATASET = "alad"
WEIGHT_FLOOR = 1e-3
JITTER = 0.22
TEST_Y = 0.08  # significance mark height above the highest AUROC, in AUROC ranges
TEST_HEADROOM = 0.1  # space above the marks, as a fraction of the axes
POINT_SIZE_MAX = 8.4
BACKGROUND_ALPHA = 0.5  # non-top ligands, so the boxes show through
AXES = ("rect",)


def plot(axs, dataset: str = DATASET):
    (ax,) = axs
    scores = common.load_runner_output(
        DATASETS[dataset].nichenet_scores_filename(),
        VALIDATION_DIR,
        hint=f"    uv run cccrise-nichenet --dataset {dataset}",
    ).dropna(subset=["auroc"])
    tests = top_ligand_tests(scores).set_index("component")
    components = sorted(scores["component"].unique())
    high = scores["auroc"].max()
    test_y = high + TEST_Y * (high - scores["auroc"].min())

    rng = np.random.default_rng(0)
    for x, component in enumerate(components):
        points = _component_points(scores[scores["component"] == component], x, ax, rng)
        significant = tests.loc[component, "q"] < common.SIG_ALPHA
        ax.text(
            x,
            test_y,
            "*" if significant else "n.s.",
            ha="center",
            va="bottom",
            fontsize=style.FONT_SIZE_DENSE,
            fontweight="bold" if significant else "normal",
            color=style.COLORS["ink"] if significant else style.COLORS["reference"],
        )
    ax.boxplot(
        [scores.loc[scores["component"] == c, "auroc"] for c in components],
        positions=range(len(components)),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        zorder=1,
        boxprops={
            "facecolor": "0.95",
            "edgecolor": style.COLORS["reference"],
            "lw": style.LINE_WIDTH_THIN,
        },
        medianprops={"color": style.COLORS["ink"], "lw": style.LINE_WIDTH},
        whiskerprops={"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
        capprops={"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
    )
    common.reference_line(ax, 0.5)
    ax.set_xticks(range(len(components)), components)
    ax.set_xlim(-0.6, len(components) - 0.4)
    # Room for the significance marks; ticks only over the AUROCs
    ax.set_ylim(top=test_y + TEST_HEADROOM * (test_y - ax.get_ylim()[0]))
    limits = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if t <= high])
    ax.set_ylim(limits)
    ax.set_xlabel("CP component")
    ax.set_ylabel("NicheNet target AUROC")
    common.colorbar(ax, points, "Relative ligand weight")


def _component_points(component_scores, x, ax, rng):
    """Weight-colored points for one component's ligands, the non-top ones
    translucent; returns the (opaque) top ligands' points."""
    c = component_scores.assign(
        w=np.clip(component_scores["rel_weight"], WEIGHT_FLOOR, 1)
    ).sort_values("w")
    c["x"] = x + rng.uniform(-JITTER, JITTER, len(c))
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


def makeFigure():
    return common.standalone_fixed(__name__)
