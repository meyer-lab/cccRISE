"""Figure 4's panels that place every component of every fit on one axis
(`_layout`): co-variation, depth tracking, what each LR column is,
outcome associations, and LR-loading shares."""

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .. import common, style
from ._common import COMPARATORS, METHOD_COLORS
from ._data import CLASSES, HOMOPHILIC, SINGLE_PAIR, beyond, both, classified, short
from ._layout import (
    BLANK_HANDLE,
    LABELLING_HANDLES,
    LABELLING_MARKERS,
    NULL_HANDLE,
    SLOTS,
    UNTESTABLE_HANDLE,
    UNTESTABLE_Y,
    fit_axis,
    label_above,
    label_sides,
    method_handles,
    spread_slots,
    strips,
)

SINGLE_HATCH = "////"
#: Figure 4B and C's keys sit this far (axes fraction) off the bottom.
KEY_RAISE = 0.07
#: Main Figure 4B and C's fits: no PB.
MAIN_SLOTS = [slot for slot in SLOTS if slot[0] != "PB"]
#: Short names of the ALAD associations.
ASSOCIATION_NAMES = {
    "Control vs. ALAD (All)": "control vs. ALAD",
    "ALAD Status (3-Way)": "ALAD status",
    "6-Month Outcome (4-Way)": "6-mo outcome",
    "6-Month CLAD vs. No CLAD": "6-mo CLAD",
    "6-Month Died vs. Alive": "6-mo death",
    "1-Year Outcome (4-Way)": "1-yr outcome",
    "1-Year CLAD vs. No CLAD": "1-yr CLAD",
    "1-Year Died vs. Alive": "1-yr death",
    "Acute Rejection (Yes vs. No)": "acute rejection",
}
OUTCOME = "6-Month Outcome (4-Way)"


def covariation_tables(disjoint: bool) -> dict[str, pd.DataFrame]:
    """Each labelling's co-variation, single-pair columns (nothing for their
    top pair to co-vary with) left out, with this version's (`disjoint`:
    pairs of pairs sharing no gene) ρ̄ as Observed, its stricter null as
    Threshold, whether it is above (Beyond) and whether it is above in
    either version (Either)."""
    sfx = "_disjoint" if disjoint else ""
    return {
        lab: t[(t["TopShare"] < SINGLE_PAIR) & (t["PairsUsed"] >= 2)].assign(
            Observed=t[f"Observed{sfx}"],
            Threshold=t[[f"random_q95{sfx}", f"expression_matched_q95{sfx}"]].max(
                axis=1
            ),
            Beyond=beyond(t, sfx),
            Either=beyond(t) | beyond(t, "_disjoint"),
        )
        for lab, t in both("covariation").items()
    }


def _ylabel(disjoint: bool) -> str:
    return (
        "Gene-disjoint co-variation, " if disjoint else "Co-variation, "
    ) + r"$\bar{\rho}$"


def _covariation_axes(
    ax, t, centres, disjoint: bool, half_width: float, labelled: bool = True
) -> None:
    """Each component's null tick (`half_width` either side) joined by a
    line to its point, an x for each without gene-disjoint pairs, the fit
    axis (`fit_axis`, `labelled`), limits and y label."""
    ax.hlines(
        t["Threshold"],
        t["X"] - half_width,
        t["X"] + half_width,
        color=style.COLORS["reference"],
        lw=0.9,
        zorder=2,
    )
    ax.vlines(t["X"], t["Threshold"], t["Observed"], color="0.45", lw=0.7, zorder=1)
    _components(ax, t)
    for method, g in t[t["Observed"].isna()].groupby("Method"):
        ax.scatter(
            g["X"],
            np.full(len(g), UNTESTABLE_Y),
            marker="x",
            s=style.POINT_SIZE,
            color=METHOD_COLORS[method],
            lw=0.7,
            zorder=3,
        )
    fit_axis(ax, centres, labelled=labelled)
    ax.set(
        xlim=(t["X"].min() - 1, t["X"].max() + 1),
        ylim=(0, 1.05),
        ylabel=_ylabel(disjoint),
    )


def _hits(t, y) -> pd.DataFrame:
    """The components above their null, with label text and height `y`."""
    hits = t[t["Beyond"]]
    return hits.assign(
        Y=y(hits),
        Text=[
            short(m, c) for m, c in zip(hits["Method"], hits["Component"], strict=True)
        ],
    )


def _components(ax, t) -> None:
    for (method, lab), g in t.groupby(["Method", "Labelling"]):
        color = METHOD_COLORS[method]
        ax.scatter(
            g["X"],
            g["Observed"],
            marker=LABELLING_MARKERS[lab],
            s=style.POINT_SIZE * 1.2,
            facecolor=np.where(g["Beyond"], color, "white"),
            edgecolor=color,
            lw=0.8,
            zorder=3,
        )


def covariation_main(axs, disjoint: bool = False, key: bool = False) -> None:
    """The components above the stricter null in either version (no PB),
    each at its own x, its null's 95th percentile a tick joined by a line to
    its ρ̄; solid, and labelled, when above the null in this version, an x
    when it has no gene-disjoint pairs. `key` adds the method (PB too, for
    the panels after) and labelling keys to the tick's."""
    ax = axs[0]
    tables = {lab: t[t["Either"]] for lab, t in covariation_tables(disjoint).items()}
    t, centres = spread_slots(tables, MAIN_SLOTS)
    _covariation_axes(ax, t, centres, disjoint, 0.38)
    label_above(ax, _hits(t, lambda h: h["Observed"]), stagger=True)
    ticks = [NULL_HANDLE, UNTESTABLE_HANDLE] if disjoint else [NULL_HANDLE]
    keys = [(ticks, "lower left", (0, KEY_RAISE), 1)]
    if key:
        methods = method_handles(("cccRISE", *COMPARATORS))
        labellings = LABELLING_HANDLES + [BLANK_HANDLE] * (
            len(methods) - len(LABELLING_HANDLES)
        )
        keys.append((methods + labellings, "lower right", (1, KEY_RAISE), 2))
    for handles, loc, anchor, ncol in keys:
        ax.add_artist(
            ax.legend(
                handles=handles,
                loc=loc,
                bbox_to_anchor=anchor,
                ncol=ncol,
                fontsize=style.FONT_SIZE_DENSE - 1,
            )
        )


def _dividers(ax, t) -> None:
    """Dotted lines between fits, solid between methods."""
    ends = (
        t.groupby(["Method", "Labelling"], sort=False)["X"]
        .agg(["min", "max"])
        .reset_index()
    )
    for left, right in zip(
        ends.iloc[:-1].itertuples(), ends.iloc[1:].itertuples(), strict=True
    ):
        new_method = left.Method != right.Method
        ax.axvline(
            (left.max + right.min) / 2,
            color="0.6" if new_method else "0.85",
            lw=0.6 if new_method else 0.5,
            ls="-" if new_method else ":",
            zorder=0,
        )


def pairs_used(axs) -> None:
    """How many top pairs each component of every fit tests, at its
    `covariation_spread` x (bare ticks: the panels below name them); the
    figure's key above, left."""
    ax = axs[0]
    t, centres = spread_slots(covariation_tables(False))
    for method, g in t.groupby("Method"):
        ax.bar(g["X"], g["PairsUsed"], width=0.7, color=METHOD_COLORS[method], lw=0)
    _dividers(ax, t)
    fit_axis(ax, centres, labelled=False)
    ax.set(
        xlim=(t["X"].min() - 1, t["X"].max() + 1),
        ylim=(0, 10.5),
        yticks=[2, 5, 10],
        ylabel="Pairs used",
    )
    ax.legend(
        handles=[
            *method_handles(("cccRISE", *COMPARATORS), 3.5),
            NULL_HANDLE,
            UNTESTABLE_HANDLE,
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncol=6,
        fontsize=style.FONT_SIZE_DENSE - 1,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


def covariation_spread(axs, disjoint: bool = False) -> None:
    """Every component of every fit at its own x, as `covariation_main`
    draws them; the x axis named only under the gene-disjoint panel, the
    bottom one. `pairs_used` carries the key."""
    ax = axs[0]
    t, centres = spread_slots(covariation_tables(disjoint))
    _covariation_axes(ax, t, centres, disjoint, 0.4, labelled=disjoint)
    _dividers(ax, t)
    label_above(ax, _hits(t, lambda h: np.maximum(h["Observed"], h["Threshold"])))
    low = min(0.0, float(t["Observed"].min()))
    # Headroom for labels at ρ̄ near 1; below 0 only for a negative ρ̄
    ax.set(ylim=(low - 0.06 if low < 0 else 0, 1.17), yticks=[0, 0.5, 1])
    if low < 0:
        common.reference_line(ax, 0)


def depth(axs) -> None:
    """|ρ| of each component's |sample profile| with depth, each fit's
    spread as a box plot behind its points."""
    ax = axs[0]
    tables = both("depth")
    strips(ax, tables, "AbsRho")
    for method, lab, x in SLOTS:
        values = tables[lab].loc[tables[lab]["Method"] == method, "AbsRho"]
        ax.boxplot(
            values,
            positions=[x],
            widths=0.34 if method == "cccRISE" else 0.3,
            showfliers=False,
            patch_artist=True,
            zorder=1,
            manage_ticks=False,
            boxprops={"facecolor": "none", "edgecolor": "0.35", "lw": 0.6},
            medianprops={"color": "0.1", "lw": 1.0},
            whiskerprops={"color": "0.35", "lw": 0.6},
            capprops={"color": "0.35", "lw": 0.6},
        )
    ax.set(ylabel="Depth tracking, |ρ|", ylim=(0, 1))


def _class_blocks(ax, g, x, width=0.4, gap=0.15) -> None:
    """One fit's stacked class blocks at `x`, single-pair columns hatched."""
    counts = g["Class"].value_counts().reindex(list(CLASSES), fill_value=0)
    single = (
        g[g["SinglePair"]]["Class"].value_counts().reindex(list(CLASSES), fill_value=0)
    )
    bottom = 0
    for name, color in CLASSES.items():
        for n, hatch in (
            (counts[name] - single[name], None),
            (single[name], SINGLE_HATCH),
        ):
            if n:
                ax.bar(
                    x,
                    n - gap,
                    width,
                    bottom=bottom + gap / 2,
                    color=color,
                    hatch=hatch,
                    edgecolor=style.COLORS["ink"],
                    hatchcolor=style.COLORS["ink"],
                    lw=0.5,
                )
            bottom += n


def classes(axs) -> None:
    """What each LR column is (`_data.CLASSES`), one stacked bar per fit."""
    ax = axs[0]
    t = classified()
    for m, lab, x in SLOTS:
        _class_blocks(ax, t[(t["Method"] == m) & (t["Labelling"] == lab)], x)
    fit_axis(ax, SLOTS)
    ax.set_ylabel("LR columns")
    handles = [
        Patch(facecolor=c, edgecolor=style.COLORS["ink"], lw=0.5, label=n)
        for n, c in CLASSES.items()
    ]
    handles.append(
        Patch(
            facecolor="white",
            edgecolor=style.COLORS["ink"],
            hatch=SINGLE_HATCH,
            lw=0.5,
            label="single-pair",
        )
    )
    ax.legend(handles=handles, loc="upper left", fontsize=style.FONT_SIZE_DENSE - 1)


def association_strip(
    axs, association: str = OUTCOME, above: float | None = None
) -> None:
    """Each component's η² for `association` (Figure 3's test), filled, and
    labelled "Xx (FDR)" in bold, where FDR < `common.SIG_ALPHA`; with
    `above`, every component of greater η² labelled too."""
    ax = axs[0]
    tables = {
        lab: t[t["Association"] == association].assign(
            Significant=t["PAdj"] < common.SIG_ALPHA
        )
        for lab, t in both("associations").items()
    }
    drawn = strips(ax, tables, "EtaSquared", "Significant")
    ax.set(
        ylabel=ASSOCIATION_NAMES[association].capitalize() + r" $\eta^2$",
        ylim=(0, drawn["EtaSquared"].max() * 1.3),
    )
    shown = drawn[
        drawn["Significant"]
        | (drawn["EtaSquared"] > (np.inf if above is None else above))
    ]
    text = [
        f"{short(m, c)} ({q:.2g})"
        for m, c, q in zip(shown["Method"], shown["Component"], shown["PAdj"])
    ]
    label_sides(
        ax,
        shown.assign(Y=shown["EtaSquared"], Bold=shown["Significant"], Text=text),
        obstacles=drawn[["X", "EtaSquared"]],
    )


def _share(axs, column: str, flag: str, ylabel: str, threshold: float) -> None:
    """Each component's `column` of `classified`, the columns where `flag`
    is true labelled, `threshold` dashed."""
    ax = axs[0]
    t = classified()
    rng = np.random.default_rng(0)
    drawn = []
    for method, lab, x in SLOTS:
        g = t[(t["Method"] == method) & (t["Labelling"] == lab)]
        xs = x + rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(
            xs,
            g[column],
            s=style.POINT_SIZE,
            marker=LABELLING_MARKERS[lab],
            color=METHOD_COLORS[method],
            edgecolor="white",
            lw=0.3,
            zorder=3,
        )
        drawn.append(
            g.assign(
                X=xs,
                Y=g[column],
                Bold=False,
                Text=[short(method, c) for c in g["Component"]],
            )
        )
    drawn = pd.concat(drawn, ignore_index=True)
    ax.axhline(
        threshold,
        color=style.COLORS["reference"],
        ls=style.LINE_STYLE_REFERENCE,
        lw=0.6,
    )
    fit_axis(ax, SLOTS)
    ax.set(ylabel=ylabel, ylim=(-0.02, 1.02))
    label_sides(
        ax,
        drawn[drawn[flag] & (drawn["Method"] != "cccRISE")],
        size=style.FONT_SIZE_DENSE - 1,
        overflow=10,
        obstacles=drawn[["X", "Y"]],
    )


def homophilic_share(axs) -> None:
    """Each component's share of squared LR loading on homophilic pairs;
    the columns `classes` calls homophilic labelled."""
    _share(
        axs,
        "HomophilicShare",
        "IsHomophilic",
        "Homophilic share of\nsquared LR loading",
        HOMOPHILIC,
    )


def top_share(axs) -> None:
    """Each component's top pair's share of squared LR loading; the
    single-pair columns `classes` hatches labelled."""
    _share(
        axs,
        "TopShare",
        "SinglePair",
        "Top pair's share of\nsquared LR loading",
        SINGLE_PAIR,
    )
