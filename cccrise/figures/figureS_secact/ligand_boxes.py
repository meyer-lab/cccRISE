"""Figure 2's NicheNet box panel (`figure2/nichenet.py` in the
fig2-fig3-cleanup worktree) under each prior: this project's pipeline
(Wilcoxon receiver ranking, top-250-target AUROC), each prior's top ligands
drawn from its own scorable ligands. Points are colored and sized by the
ligand's weight in the component, opaque for the top ligands and translucent
for the rest; above each box, a star where the two-sided Mann-Whitney test
of the top ligands against all others has BH q below `SIG_ALPHA`, else n.s.
Both priors share the AUROC axis."""

from functools import partial
from types import SimpleNamespace

import numpy as np
from matplotlib.colors import LogNorm

from ...validation.nichenet import top_ligand_tests
from .. import common, style
from . import _common

WEIGHT_FLOOR = 1e-3
JITTER = 0.22
TEST_Y = 0.08  # significance mark height above the highest AUROC, in AUROC ranges
TEST_HEADROOM = 0.1  # space above the marks, as a fraction of the axes
POINT_SIZE_MAX = 8.4
BACKGROUND_ALPHA = 0.5  # non-top ligands, so the boxes show through


def plot(fig, prior: str):
    ours = (
        _common.load()
        .query("universe == 'own' and profile == 'wilcoxon' and scoring == 'auroc'")
        .dropna(subset=["score"])
        .rename(columns={"score": "auroc"})
    )
    low, high = ours["auroc"].min(), ours["auroc"].max()
    test_y = high + TEST_Y * (high - low)
    scores = ours[ours["prior"] == prior]
    tests = top_ligand_tests(scores).set_index("component")
    components = sorted(scores["component"].unique())

    ax = fig.subplots()
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
    # Shared range across priors; room for the significance marks, ticks only
    # over the AUROCs
    bottom = low - 0.05 * (high - low)
    ax.set_ylim(bottom, test_y + TEST_HEADROOM * (test_y - bottom))
    limits = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if t <= high])
    ax.set_ylim(limits)
    ax.set_xlabel("CP component")
    ax.set_ylabel(f"{_common.PRIOR_LABEL[prior]} target AUROC")
    ax.set_title(
        f"{_common.PRIOR_LABEL[prior]} ({len(scores) // len(components)} ligands)"
    )
    fig.colorbar(points, ax=ax, label="Relative ligand weight", shrink=0.8, pad=0.02)


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


NICHENET = SimpleNamespace(plot=partial(plot, prior="nichenet"))
SECACT = SimpleNamespace(plot=partial(plot, prior="secact"))


def makeFigure():
    return common.standalone(NICHENET.plot, style.PANEL_SIZE["half"])
