"""Figure 4's panels that place every component of every fit on one axis
(`_layout`): co-variation, depth tracking, what each LR column is,
outcome associations, and LR-loading shares."""

import numpy as np
import pandas as pd

from ... import covariation
from .. import common, style
from ._common import (
    COMPARATORS,
    DATASET,
    DATASET_LABELS,
    METHOD_COLORS,
)
from ._data import (
    HOMOPHILIC,
    SINGLE_PAIR,
    beyond,
    both,
    classified,
    short,
    source_data,
)
from ._layout import (
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
from ._recovery import panel_title

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


def covariation_tables(
    disjoint: bool, dataset: str = DATASET
) -> dict[str, pd.DataFrame]:
    """Each labelling's co-variation of the components above the stricter
    null in either version (single-pair columns, with nothing for their top
    pair to co-vary with, left out), with this version's (`disjoint`: pairs of
    pairs sharing no gene) ρ̄ as Observed, its stricter null as Threshold and
    whether it is above (Beyond)."""
    sfx = "_disjoint" if disjoint else ""
    return {
        lab: t[
            (t["TopShare"] < SINGLE_PAIR)
            & (t["PairsUsed"] >= 2)
            & (beyond(t) | beyond(t, "_disjoint"))
        ].assign(
            Observed=t[f"Observed{sfx}"],
            Threshold=t[[f"random_q95{sfx}", f"expression_matched_q95{sfx}"]].max(
                axis=1
            ),
            Beyond=beyond(t, sfx),
        )
        for lab, t in both("covariation", dataset).items()
    }


def _covariation_source(t, disjoint: bool) -> None:
    """The components `t` shows (`covariation_tables`), as source data."""
    sfx = "_disjoint" if disjoint else ""
    pairs = "gene-disjoint pairs" if disjoint else "pairs"
    source_data(
        ("gene_disjoint_" if disjoint else "") + "top_pair_covariation_by_component",
        t,
        {
            "PairsUsed": "Top pairs",
            "TopShare": "Top pair share",
            "Observed": "Mean rho",
            f"random_q95{sfx}": "Random null 95th pct.",
            f"expression_matched_q95{sfx}": "Expression-matched null 95th pct.",
            "Threshold": "Stricter null 95th pct.",
            "Beyond": "Above null",
        },
        "Each component's top LR pairs (the fewest carrying "
        f"{covariation.CARRIED_SHARE:.0%} of its squared LR loading, at most "
        f"{covariation.N_TOP}; Top pairs), each scored per sample in the "
        "component's own sender/receiver context on its method's tensor: their "
        f"mean pairwise Spearman correlation across samples over {pairs} of "
        "them (Mean rho; empty where no pair of pairs is gene-disjoint, an x), "
        "against the 95th percentiles of the same mean over random and "
        "expression-matched draws of LR pairs, and whether it is above the "
        "stricter (filled). Top pair share is the top pair's share of squared "
        f"LR loading; columns of {SINGLE_PAIR} or more are left out.",
    )


def _ylabel(disjoint: bool) -> str:
    return r"$\bar{\rho}$, " + ("gene-disjoint" if disjoint else "all pairs")


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
            color=METHOD_COLORS[str(method)],
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


def pairs_used(axs, dataset: str = DATASET, key: bool = False) -> None:
    """How many top pairs each component of every fit tests, at its
    `covariation_spread` x (bare ticks: the panels below name them), under
    the cohort's name; with `key`, the figure's key above it."""
    ax = axs[0]
    t, centres = spread_slots(covariation_tables(False, dataset))
    source_data(
        "top_pairs_tested_by_component",
        t,
        {"PairsUsed": "Top pairs"},
        "How many top LR pairs each component's co-variation tests: the fewest "
        f"carrying {covariation.CARRIED_SHARE:.0%} of its squared LR loading, "
        f"at most {covariation.N_TOP}; columns whose top pair carries "
        f"{SINGLE_PAIR} or more are left out.",
    )
    for method, g in t.groupby("Method"):
        ax.bar(
            g["X"], g["PairsUsed"], width=0.7, color=METHOD_COLORS[str(method)], lw=0
        )
    _dividers(ax, t)
    fit_axis(ax, centres, labelled=False)
    ax.set(
        xlim=(t["X"].min() - 1, t["X"].max() + 1),
        ylim=(0, 10.5),
        yticks=[2, 5, 10],
        ylabel="Pairs used",
    )
    ax.yaxis.label.set_size(style.FONT_SIZE_DENSE)
    ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    panel_title(ax, DATASET_LABELS[dataset])
    if not key:
        return
    ax.legend(
        handles=[
            *method_handles(("cccRISE", *COMPARATORS), 3.5),
            NULL_HANDLE,
            UNTESTABLE_HANDLE,
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.42),
        ncol=6,
        fontsize=style.FONT_SIZE_DENSE - 1,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


def covariation_spread(axs, disjoint: bool = False, dataset: str = DATASET) -> None:
    """Every component above the null of every fit at its own x, as
    `_covariation_axes` draws them; the x axis named only under the
    gene-disjoint panel, the bottom one. `pairs_used` carries the key."""
    ax = axs[0]
    t, centres = spread_slots(covariation_tables(disjoint, dataset))
    _covariation_source(t, disjoint)
    _covariation_axes(ax, t, centres, disjoint, 0.4, labelled=disjoint)
    _dividers(ax, t)
    label_above(ax, _hits(t, lambda h: np.maximum(h["Observed"], h["Threshold"])))
    low = min(0.0, float(t["Observed"].min()))
    # Headroom for labels at ρ̄ near 1; below 0 only for a negative ρ̄
    ax.set(ylim=(low - 0.06 if low < 0 else 0, 1.17), yticks=[0, 0.5, 1])
    if low < 0:
        common.reference_line(ax, 0)
    ax.yaxis.label.set_size(style.FONT_SIZE_DENSE)
    ax.tick_params(labelsize=style.FONT_SIZE_DENSE)


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
    source_data(
        "association_eta_squared_by_component",
        drawn,
        {
            "Association": "Association",
            "EtaSquared": "Eta squared",
            "P": "p",
            "PAdj": "FDR",
            "Significant": "Significant",
        },
        f"Each component's association with '{association}': one-way ANOVA of "
        "its read-depth-corrected sample profile over the association's "
        "groups (eta squared), Benjamini-Hochberg adjusted across the fit's "
        f"associations and components (FDR); Significant (filled) below FDR "
        f"{common.SIG_ALPHA}.",
    )
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


def _share(
    axs, column: str, flag: str, ylabel: str, threshold: float, source: dict
) -> None:
    """Each component's `column` of `classified`, the columns where `flag`
    is true labelled, `threshold` dashed; saved as source data by `source`'s
    name and description."""
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
    source_data(
        source["name"],
        drawn,
        {column: ylabel.replace("\n", " "), flag: flag},
        source["description"],
    )
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
        {
            "name": "homophilic_share_by_component",
            "description": (
                "Each component's share of squared LR loading on homophilic pairs "
                "(a ligand whose genes are its receptor's); IsHomophilic (labelled) "
                f"at {HOMOPHILIC} or more, unless the column is associated with an "
                "outcome or contained in cccRISE's LR span (`lr_column_classes`)."
            ),
        },
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
        {
            "name": "top_pair_share_by_component",
            "description": (
                "Each component's top LR pair's share of its squared LR loading; "
                f"SinglePair (labelled) at {SINGLE_PAIR} or more."
            ),
        },
    )
