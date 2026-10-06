"""Each comparator fit's strongest clinical association (best component, any
of the 13 associations, maxT permutation p) on the raw, winsorised and
block-normed tensor; cccRISE's for reference."""

import numpy as np

from .. import common, style
from ._common import METHODS, SHORT, VARIANT_LABELS, load


def plot(fig):
    null = load("benchmark_anova_null")
    best = null.groupby(["variant", "Method", "RankMode"])["PermP"].min()
    ax = fig.subplots()
    x = np.arange(len(VARIANT_LABELS))
    for i, method in enumerate(METHODS):
        for rank_mode, marker in (("own", "o"), ("shared", "s")):
            y = [best.get((v, method, rank_mode), np.nan) for v in VARIANT_LABELS]
            ax.plot(
                x + (i - 1) * 0.12,
                -np.log10(y),
                marker=marker,
                ls="none",
                color=style.CATEGORICAL[i + 1],
                markerfacecolor=style.CATEGORICAL[i + 1]
                if rank_mode == "own"
                else "white",
                label=f"{SHORT[method]} {'own' if rank_mode == 'own' else 'rank 5'}",
            )
    ours = best.get(("none", "cccRISE", "published"))
    ax.axhline(
        -np.log10(ours), color=style.COLORS["emphasis"], lw=1, label="cccRISE C4"
    )
    ax.axhline(-np.log10(0.05), color=style.COLORS["reference"], ls="--", lw=0.8)
    ax.set_xticks(x, list(VARIANT_LABELS.values()))
    ax.set_ylabel(r"$-\log_{10}$ best maxT p")
    ax.legend(
        fontsize=style.FONT_SIZE_DENSE - 1.5, frameon=False, ncols=2, loc="upper left"
    )
    ax.grid(False)


def makeFigure():
    return common.standalone(plot, style.PANEL_SIZE["half"])
