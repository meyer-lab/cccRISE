"""Mean pairwise rank correlation across samples of the carrying LR pairs of
every ALAD CCC-RISE component and every PB and STAC LR column, against random
and expression-matched null draws.

Figure 5D's co-variation test (`run_comparator_scores`' ``covariation``) on
every component and column, grouped by fit (TC2C left out, as Figure 5G): each
one's observed co-variation over both nulls' 95th percentiles (random dashed,
expression-matched solid). Filled where its one-sided empirical p against the
expression-matched null, BH-corrected within its fit, is below
`common.SIG_ALPHA` and it lies above both nulls' 95th percentiles; crossed
where its carrying pairs merge into fewer than two gene groups (untested)."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import false_discovery_control

from ... import common, style
from .._common import METHOD_COLORS, load
from .._data import LABELLINGS
from .._split import NULL_TICKS, _lollipop

AXES = ("split_covariation",)
#: CCC-RISE, then each PB and STAC fit, each named under its block.
SLOTS = [("cccRISE", "coarse", "CCC-RISE")] + [
    (m, lab, f"{m} {lab[0].upper()}") for m in ("PB", "STAC") for lab in LABELLINGS
]
SLOT_GAP = 0.7
#: The y axis's top: room for the key above every point (ticks stop at 1)
KEY_TOP = 1.05
#: Column numbers (each block's name gives its fit): horizontal, at this size (pt)
TICK_ROTATION = 0
TICK_SIZE = style.FONT_SIZE_LABEL
FIT = ["Method", "Labelling", "Component"]


def covariation() -> pd.DataFrame:
    """Every CCC-RISE component's and PB/STAC column's co-variation, with its
    expression-matched q (BH within its fit, over the tested rows) and whether
    it is drawn filled (q < `common.SIG_ALPHA`, above both nulls)."""
    t = load("covariation", dataset="alad")
    # cccRISE's components don't depend on the labelling: its coarse rows
    cccrise = (t["Method"] == "cccRISE") & (t["Labelling"] == "coarse")
    t = t[t["Method"].isin(["PB", "STAC"]) | cccrise].copy()
    q = pd.Series(np.nan, index=t.index)
    tested = t[t["Observed"].notna()]
    for _, g in tested.groupby(["Method", "Labelling"]):
        q[g.index] = false_discovery_control(g["expression_matched_p"])
    above = (t["Observed"] > t["random_q95"]) & (
        t["Observed"] > t["expression_matched_q95"]
    )
    return t.assign(
        expression_matched_q=q, Filled=(q < common.SIG_ALPHA) & above
    ).reset_index(drop=True)


def null_draw_means(t: pd.DataFrame) -> pd.DataFrame:
    """Each null draw's mean pairwise correlation (`run_comparator_scores`'
    ``covariation_draws``) for `t`'s rows: the points behind each null's 95th
    percentile and p."""
    draws = load("covariation_draws", dataset="alad").merge(t[FIT])
    return (
        draws.groupby([*FIT, "Null", "Draw"])["Spearman correlation"]
        .mean()
        .rename("Mean pairwise correlation")
        .reset_index()
    )


def _source(t: pd.DataFrame) -> None:
    # cccRISE's rows unlabelled, as Figure 5D's (its components have no labelling)
    def unlabelled(frame):
        return frame.assign(
            Labelling=frame["Labelling"].where(frame["Method"] != "cccRISE", "")
        )

    common.source_data(
        "component_and_column_covariation_alad",
        unlabelled(t)[
            [
                *FIT,
                "PairsUsed",
                "GeneGroups",
                "Observed",
                "random_q95",
                "expression_matched_q95",
                "random_p",
                "expression_matched_p",
                "expression_matched_q",
                "Filled",
            ]
        ],
        "Mean pairwise rank correlation across samples of each CCC-RISE component's "
        "and PB and STAC LR column's carrying pairs, pairs sharing a gene merged into "
        "one (GeneGroups of them), with the 95th percentiles of the random and of the "
        "expression-matched null, merged alike, and each null's one-sided empirical p "
        "((1 + draws at or above the observed) / (1 + draws)); the expression-matched "
        "p's Benjamini-Hochberg q within each fit's tested rows; NaN where fewer than "
        "two groups remain. Filled: q < 0.05 and Observed above both 95th "
        "percentiles.",
    )
    common.source_data(
        "covariation_null_draw_means_alad",
        unlabelled(null_draw_means(t)),
        "Each null draw's mean pairwise rank correlation (Spearman, across samples) "
        "for every component and column of the co-variation file: the points behind "
        "its nulls' 95th percentiles and p's.",
    )


def _untested(ax, x: float, method: str) -> None:
    ax.scatter(x, 0.03, marker="x", s=8, lw=0.6, color=METHOD_COLORS[method], zorder=3)


def _draw(ax, t: pd.DataFrame) -> tuple[list, list, list]:
    """Each fit's block of points, `SLOT_GAP` apart: the ticks, their names, and
    each block's (first x, last x, name)."""
    x, ticks, names, blocks = 0.0, [], [], []
    for method, lab, name in SLOTS:
        g = t[(t["Method"] == method) & (t["Labelling"] == lab)]
        first = x
        for _, r in g.sort_values("Component").iterrows():
            if np.isnan(r["Observed"]):
                _untested(ax, x, method)
            else:  # `_lollipop` fills on q: hand it this panel's fill
                r = r.copy()
                r["X"], r["expression_matched_q"] = x, 0.0 if r["Filled"] else 1.0
                _lollipop(ax, r)
            ticks.append(x)
            names.append(str(int(r["Component"])))
            x += 1
        blocks.append((first, x - 1, name))
        x += SLOT_GAP
    return ticks, names, blocks


def _name_blocks(ax, blocks: list) -> None:
    """Each fit's name centred under its block, below the column names."""
    renderer = ax.figure.canvas.get_renderer()
    bottom = min(t.get_window_extent(renderer).y0 for t in ax.get_xticklabels())
    y = ax.transAxes.inverted().transform((0, bottom))[1] - 0.02
    for first, last, name in blocks:
        ax.text(
            (first + last) / 2,
            y,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=style.FONT_SIZE_LABEL,
        )


def _key(ax) -> None:
    """Two rows at the top left: the nulls' ticks and the untested cross, then
    filled and hollow points under the ticks (the legend fills by column)."""
    random, matched = (Line2D([], [], **tick) for tick in NULL_TICKS.values())
    filled, hollow = (
        Line2D([], [], ls="", marker="o", ms=3.8, mec="0.4", mfc=fill, label=label)
        for fill, label in (
            ("0.4", f"q < {common.SIG_ALPHA}"),
            ("white", f"q ≥ {common.SIG_ALPHA}"),
        )
    )
    untested = Line2D(
        [], [], ls="", marker="x", ms=3.6, mew=0.7, color="0.4", label="< 2 gene groups"
    )
    ax.legend(
        handles=[random, filled, matched, hollow, untested],
        loc="upper left",
        ncol=3,
        fontsize=style.FONT_SIZE_LABEL,
        handlelength=1.6,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.6,
        frameon=False,
    )


def plot(axs):
    (ax,) = axs
    t = covariation()
    _source(t)
    ticks, names, blocks = _draw(ax, t)
    ax.set(
        ylim=(min(0, t["Observed"].min() - 0.05), KEY_TOP),
        yticks=[0, 0.5, 1],
        ylabel="Mean pairwise correlation\nacross conditions",
        xlim=(-0.8, ticks[-1] + 0.8),
    )
    ax.set_xticks(ticks, names)
    ax.tick_params(axis="x", labelrotation=TICK_ROTATION, labelsize=TICK_SIZE, pad=1.5)
    ax.figure.canvas.draw()  # the column names' extent, to set the blocks' under
    _name_blocks(ax, blocks)
    _key(ax)


def makeFigure():
    return common.standalone_fixed(__name__)
