"""NicheNet downstream-target validation of CPD components, from
``uv run cccrise-nichenet``: each component's top ligands against every other
ligand the model sees.

One box per component: the NicheNet-style AUROC
(``nichenet.component_ligand_scores``) of every scored ligand on the
component's receiver ranking, restricted to genes expressed in the receiver
groups. Points are colored and sized by the ligand's weight in the component;
the top ligands (``nichenet.top_pair_ligands``) are outlined and labeled.
Above each box: a two-sided Mann-Whitney test of the top ligands against all
the others, and its Benjamini-Hochberg q across components
(``nichenet.top_ligand_tests``)."""

import numpy as np
from matplotlib.colors import LogNorm

from ...runners.datasets import DATASETS, VALIDATION_DIR
from ...validation.nichenet import top_ligand_tests
from .. import common, style

DATASET = "alad"
WEIGHT_FLOOR = 1e-3
JITTER = 0.22
LABEL_X = 0.3  # label offset from the box center, in component units
LABEL_GAP = 0.1  # fraction of the AUROC range between stacked labels
TEST_Y = 0.3  # p/q text height above the highest AUROC, same units
POINT_SIZE_MAX = 25


def plot(fig):
    scores = common.load_runner_output(
        DATASETS[DATASET].nichenet_scores_filename(),
        VALIDATION_DIR,
        hint=f"    uv run cccrise-nichenet --dataset {DATASET}",
    )
    plot_scores(fig, scores)


def plot_scores(fig, scores, title=None):
    """The panel for one model's `nichenet.component_ligand_scores` table;
    `title` replaces the default test description (used for comparators)."""
    scores = scores.dropna(subset=["auroc"])
    tests = top_ligand_tests(scores).set_index("component")
    components = sorted(scores["component"].unique())
    low, high = scores["auroc"].min(), scores["auroc"].max()
    span = high - low
    test_y = high + TEST_Y * span

    ax = fig.subplots()
    rng = np.random.default_rng(0)
    for x, component in enumerate(components):
        points = _component_points(
            scores[scores["component"] == component], x, LABEL_GAP * span, ax, rng
        )
        _test_text(tests.loc[component], x, test_y, ax)
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
    ax.set_xticks(
        range(len(components)),
        [
            f"{c}\n({tests.loc[c, 'n_top']:.0f} top / {tests.loc[c, 'n_comparison']:.0f})"
            for c in components
        ],
    )
    ax.set_xlim(-0.6, len(components) - 0.1)
    ax.set_ylim(top=test_y + LABEL_GAP * span)
    ax.set_xlabel("CP component")
    ax.set_ylabel("Ligand AUROC")
    ax.set_title(
        title
        or "Top ligands (outlined) vs all other ligands; "
        f"two-sided Mann-Whitney, BH q (bold q < {common.SIG_ALPHA})"
    )
    fig.colorbar(points, ax=ax, label="Relative ligand weight", shrink=0.8, pad=0.01)


def _component_points(component_scores, x, gap, ax, rng):
    """Weight-colored points for one component's ligands, top ligands labeled."""
    c = component_scores.assign(
        w=np.clip(component_scores["rel_weight"], WEIGHT_FLOOR, 1)
    ).sort_values("w")
    jitter = rng.uniform(-JITTER, JITTER, len(c))
    top = c["is_top"].to_numpy()
    points = ax.scatter(
        x + jitter,
        c["auroc"],
        c=c["w"],
        cmap=style.CMAPS["weight"],
        norm=LogNorm(vmin=WEIGHT_FLOOR, vmax=1),
        s=style.POINT_SIZE_BACKGROUND / 2 + POINT_SIZE_MAX * c["w"],
        lw=np.where(top, style.LINE_WIDTH_THIN * 1.5, 0.2),
        edgecolor=np.where(top, style.COLORS["ink"], "white"),
        zorder=3,
    )
    labeled = c[top]
    heights = _spread(labeled["auroc"].to_numpy(), gap)
    for ligand, auroc, xi, y in zip(
        labeled["ligand"], labeled["auroc"], x + jitter[top], heights, strict=True
    ):
        ax.plot(
            [xi, x + LABEL_X],
            [auroc, y],
            color=style.COLORS["reference"],
            lw=style.LINE_WIDTH_THIN * 0.8,
            zorder=2,
        )
        ax.text(
            x + LABEL_X + 0.01,
            y,
            ligand,
            fontsize=style.FONT_SIZE_DENSE,
            va="center",
            color=style.COLORS["ink"],
            zorder=4,
        )
    return points


def _spread(values, gap):
    """Label heights at least `gap` apart, in the input order."""
    order = np.argsort(values)
    placed = values[order].astype(float)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    out = np.empty_like(placed)
    out[order] = placed
    return out


def _test_text(test, x, y, ax):
    """Direction, p, and q of one component's top-vs-rest test."""
    significant = test["q"] < common.SIG_ALPHA
    arrow = "↑" if test["direction"] == "up" else "↓"
    ax.text(
        x,
        y,
        f"{arrow} p={test['p']:.2g}, q={test['q']:.2g}",
        ha="center",
        va="center",
        fontsize=style.FONT_SIZE_DENSE,
        color=style.COLORS["ink"] if significant else style.COLORS["reference"],
        fontweight="bold" if significant else "normal",
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
