"""Every method on one scale: each top ligand's percentile among its own
component's other ligands (0.5 = no better than the ligands the component
barely uses). Left: pooled over components, one-sided Mann-Whitney of cccRISE
against each comparator at its own rank with BH q (across those comparators),
and the common-language effect size.
Right: each component's mean, with a permutation test of the difference in
means against cccRISE (`validation.nichenet_comparators`)."""

import numpy as np

from ...validation.nichenet_comparators import (
    compare_methods,
    component_means,
    component_permutation_test,
    standardized_top_scores,
)
from .. import common, style
from ._common import FITS, all_scores, fit_label

JITTER = 0.18


def plot(fig):
    scores = all_scores()
    std = standardized_top_scores(scores)
    tests = compare_methods(std).set_index(["method", "rank_mode"])
    means = component_means(std)
    labels = [fit_label(m, r, scores) for m, r in FITS]
    ax_pool, ax_comp = fig.subplots(1, 2, sharey=True)
    rng = np.random.default_rng(0)
    ref_means = means.loc[means["method"] == "cccRISE", "mean_percentile"]
    for x, (method, mode) in enumerate(FITS):
        colour = (
            style.COLORS["emphasis"] if method == "cccRISE" else style.COLORS["ink"]
        )
        pooled = std.loc[
            (std["method"] == method) & (std["rank_mode"] == mode), "percentile"
        ]
        _strip(ax_pool, x, pooled.to_numpy(), colour, rng)
        comp = means.loc[
            (means["method"] == method) & (means["rank_mode"] == mode),
            "mean_percentile",
        ]
        _strip(ax_comp, x, comp.to_numpy(), colour, rng)
        if method != "cccRISE":
            t = tests.loc[(method, mode)]
            _note(ax_pool, x, f"CLES {t['cles']:.2f}\nq={t['q']:.2g}", t["q"])
            perm = component_permutation_test(ref_means, comp)
            _note(
                ax_comp, x, f"Δ {perm['difference']:+.2f}\np={perm['p']:.2g}", perm["p"]
            )
    for ax, title in (
        (ax_pool, "Top ligands, pooled (one-sided MWU vs cccRISE, BH q)"),
        (ax_comp, "Per component (permutation test vs cccRISE)"),
    ):
        common.reference_line(ax, 0.5)
        ax.set_xticks(range(len(FITS)), labels, rotation=25, ha="right")
        ax.set_ylim(-0.05, 1.25)
        ax.set_title(title)
    ax_pool.set_ylabel("Percentile among the\ncomponent's other ligands")


def _strip(ax, x, values, colour, rng):
    if not len(values):
        return
    ax.boxplot(
        values,
        positions=[x],
        widths=0.5,
        showfliers=False,
        boxprops={"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
        medianprops={"color": style.COLORS["ink"], "lw": style.LINE_WIDTH},
        whiskerprops={"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
        capprops={"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
    )
    ax.scatter(
        x + rng.uniform(-JITTER, JITTER, len(values)),
        values,
        s=style.POINT_SIZE,
        color=colour,
        zorder=3,
        lw=0,
    )


def _note(ax, x, text, p):
    significant = p < common.SIG_ALPHA
    ax.text(
        x,
        1.13,
        text,
        ha="center",
        va="center",
        fontsize=style.FONT_SIZE_DENSE,
        color=style.COLORS["ink"] if significant else style.COLORS["reference"],
        fontweight="bold" if significant else "normal",
    )


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["full"])
