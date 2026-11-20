"""LR pairs scored per condition in CPD component 5's sender/receiver context:
the eta-squared of each pair's one-way ANOVA against the 6-month outcome, with
the component's and its 10 top pairs' marked; and the Spearman correlations
across conditions among the top pairs against those among random, expression-
matched, and other outcome-associated LR pairs.

Coordinated changes: the outcome-associated CPD component against its LR
pairs scored on their own, from the raw communication tensor. Every number
is `cccrise-alad-programs`' (`run_alad_programs`); this module only reads and
draws them.

Every LR pair is scored per condition in the component's own sender/receiver
context (`program_context_scores`: the raw tensor slice for that pair,
weighted by the component's sender and receiver eigenstate loadings and
summed over eigenstates).

- Association: each pair's scores, and the component's condition loadings,
  get the component ANOVA against the 6-month outcome
  (`program_lr_association`: one-way ANOVA over the outcome groups of each
  pair's scores divided by the condition's read depth, eta^2 as the effect
  size, Benjamini-Hochberg FDR across LR pairs). The histogram
  is every LR pair's eta^2, with the whole component's eta^2 and its `N_TOP`
  most heavily weighted LR pairs' marked.
- Co-variation of the `N_TOP` pairs: the Spearman correlation across
  conditions of every pair of them (`pairwise_rank_correlations`), ranks so a
  few conditions in which many LR pairs spike together do not dominate it.
  Pairs sharing a ligand or receptor gene, which co-vary almost by
  construction, are first averaged into one (their ranks averaged), in the
  top pairs and in every draw alike. Each baseline box pools those correlations over `COVARIATION_DRAWS` draws of
  as many LR pairs from outside that set (`program_covariation`, from
  `covariation_baselines`): random,
  expression-matched (one pair per top pair from its `N_EXPRESSION_NEIGHBORS`
  nearest pairs in log mean score), and other associated (FDR < 0.05 on their own), so every box is a
  distribution of single pairwise correlations.
- Significance of the top pairs against each baseline (`covariation_tests`,
  the one statistic this module computes): the share of that baseline's draws
  whose mean correlation reaches the top pairs' mean (one-sided empirical
  p, +1 smoothed so its floor is 1 / (`COVARIATION_DRAWS` + 1), as Figure 5's
  `program_covariation` test), Benjamini-Hochberg-adjusted across baselines. Draws, not pooled
  correlations, are the unit: pairs of pairs within a draw share pairs, so a
  test over the pooled correlations would overstate n."""

from functools import cache

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from ...covariation import (
    COVARIATION_DRAWS,
    N_EXPRESSION_NEIGHBORS,
    N_TOP,
    lr_pair_genes,
    shared_gene_labels,
    top_pairs,
)
from .. import common, style
from ..figure2.nichenet import significance_marks
from ._common import (
    COMPONENT,
    OUTCOME,
    association_table,
    condition_obs,
    factors_adata,
)

BASELINES = {  # label, color
    "random": ("Random", style.COLORS["muted"]),
    "expression_matched": ("Expr.-matched", style.COLORS["receiver"]),
    "associated": ("Other associated", style.COLORS["up"]),
}
AXES = ("square", "square")
MARK_SIZE = style.FONT_SIZE_LABEL
MARK_HEADROOM = 0.22  # above the highest whisker, for its mark


def plot(axs):
    result = coordinated_analysis()
    _source_data(result)
    plot_lr_association_distribution(
        result["lr_eta"], result["component_eta"], result["top"], axs[0]
    )
    plot_lr_covariation(
        result["observed"], result["baselines"], result["tests"]["PAdj"], axs[1]
    )


def plot_lr_association_distribution(
    lr_eta: np.ndarray, component_eta: float, top: np.ndarray, ax
) -> None:
    ax.hist(lr_eta, bins=40, color=style.COLORS["muted"], label="All LR pairs")
    for i, eta in enumerate(lr_eta[top]):
        ax.axvline(
            eta,
            ymax=0.08,
            color=style.COLORS["highlight"],
            linewidth=style.LINE_WIDTH,
            label=f"Cmp. {COMPONENT} top {len(top)}" if i == 0 else None,
        )
    ax.axvline(
        component_eta,
        color=style.COLORS["emphasis"],
        linewidth=style.LINE_WIDTH,
        linestyle=style.LINE_STYLE_REFERENCE,
        label=f"Cmp. {COMPONENT}",
    )
    ax.set(xlabel=r"Outcome association ($\eta^2$)", ylabel="LR pairs")
    # Opaque, so it hides the component line where they cross
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
    ).set_zorder(5)


def plot_lr_covariation(
    observed: np.ndarray, baselines: dict[str, np.ndarray], padj, ax
) -> None:
    """Pairwise rank correlations of the top pairs (boxed, with each pair of
    pairs as a point) beside every draw's pairwise correlations, pooled, per
    baseline (NaN where a draw merged pairs), each baseline starred above its
    whisker where the top pairs exceed it (`padj`)."""
    groups = {f"Cmp. {COMPONENT} top {N_TOP}": observed} | {
        BASELINES[name][0]: draws[~np.isnan(draws)] for name, draws in baselines.items()
    }
    # Green, as the top pairs are in the association panel
    colors = [style.COLORS["highlight"]] + [BASELINES[name][1] for name in baselines]
    boxes = ax.boxplot(
        list(groups.values()),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": style.COLORS["ink"], "lw": style.LINE_WIDTH},
        whiskerprops={"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
        capprops={"color": style.COLORS["reference"], "lw": style.LINE_WIDTH_THIN},
    )
    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set(facecolor=color, alpha=0.7, edgecolor=style.COLORS["reference"])
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(observed))
    ax.scatter(
        1 + jitter,
        observed,
        color=style.COLORS["ink"],
        s=style.POINT_SIZE_BACKGROUND,
        linewidths=0,
        zorder=3,
    )
    upper_whiskers = [w.get_ydata().max() for w in boxes["whiskers"][3::2]]
    significance_marks(
        ax, range(2, len(groups) + 1), upper_whiskers, padj, fontsize=MARK_SIZE
    )
    # Room for the marks within the axes
    ax.set_ylim(top=max(ax.get_ylim()[1], max(upper_whiskers) + MARK_HEADROOM))
    common.reference_line(ax, 0)
    ax.set_xticks(
        range(1, len(groups) + 1),
        list(groups),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Pairwise correlation\nacross conditions")


@cache
def coordinated_analysis() -> dict:
    """The component's and every LR pair's outcome association, the
    component's top LR pairs, and those pairs' co-variation against
    `covariation_baselines`."""
    adata = factors_adata()
    names = np.asarray(adata.uns["lr_pairs"]).astype(str)
    scores = context_score_matrix()
    top = top_pairs(np.asarray(adata.uns["D"])[:, COMPONENT - 1])
    per_lr = (
        common.alad_programs("program_lr_association")
        .query("Component == @COMPONENT")
        .set_index("LR pair")
        .loc[names]
        .reset_index(drop=True)
    )
    observed, baselines = covariation_draws()
    table = association_table()
    component = table[
        (table["Association"] == OUTCOME) & (table["Component"] == COMPONENT)
    ]
    return {
        "scores": scores,
        "per_lr": per_lr,
        "lr_eta": per_lr.EtaSquared.to_numpy(),
        "component_eta": float(component["EtaSquared"].iloc[0]),
        "top": top,
        "observed": observed,
        "baselines": baselines,
        "tests": covariation_tests(observed, baselines),
    }


def covariation_tests(
    observed: np.ndarray, baselines: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Each baseline's one-sided empirical p of the top pairs' mean
    correlation against its per-draw means, BH-adjusted across baselines."""
    mean = observed.mean()
    draw_means = {name: np.nanmean(d, axis=1) for name, d in baselines.items()}
    p = [(1 + (m >= mean).sum()) / (1 + len(m)) for m in draw_means.values()]
    return pd.DataFrame(
        {
            "Baseline": [BASELINES[name][0] for name in baselines],
            "Top pairs mean": mean,
            "Baseline median of draw means": [
                np.median(m) for m in draw_means.values()
            ],
            "P": p,
            "PAdj": false_discovery_control(p),
        }
    )


def context_score_matrix() -> np.ndarray:
    """`COMPONENT`'s context scores, conditions (`condition_obs` order) x LR
    pairs (the factors file's order)."""
    names = np.asarray(factors_adata().uns["lr_pairs"]).astype(str)
    rows = common.alad_programs("program_context_scores").query(
        "Component == @COMPONENT"
    )
    wide = rows.pivot(index="Condition", columns="LR pair", values="Score")
    return wide.loc[condition_obs().index, names].to_numpy()


def covariation_draws() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The top pairs' pairwise rank correlations, and each baseline's draws
    (draws x pairs of pairs, NaN past a draw's merged pairs), in
    `covariation_baselines`' order, for the sets in `BASELINES`."""
    rows = common.alad_programs("program_covariation").query("Component == @COMPONENT")
    rows = rows[rows["Set"].isin(["Top pairs", *BASELINES])]
    draws = {
        str(name): group.pivot(
            index="Draw", columns="Pair of pairs", values="Rank correlation"
        ).to_numpy()
        for name, group in rows.groupby("Set", sort=False)
    }
    observed = draws.pop("Top pairs")[0]
    return observed, draws


def _source_data(result: dict) -> None:
    """Each LR pair's per-condition scores and outcome ANOVA (association
    panel), and the pairwise correlations of the top pairs and baselines
    (co-variation panel)."""
    adata = factors_adata()
    names = np.asarray(adata.uns["lr_pairs"]).astype(str)
    conditions = condition_obs().index
    common.source_data(
        "component_context_lr_scores",
        pd.DataFrame(result["scores"], index=conditions, columns=names).rename_axis(
            index="Condition"
        ),
        f"Each LR pair's raw communication score per condition in CPD component "
        f"{COMPONENT}'s sender/receiver context (the tensor slice weighted by "
        "the component's sender and receiver eigen-state loadings, summed).",
    )
    top = np.isin(np.arange(len(names)), result["top"])
    common.source_data(
        "lr_pair_outcome_anova",
        result["per_lr"][["F", "P", "PAdj", "EtaSquared"]].assign(
            **{"LR pair": names, f"Cmp {COMPONENT} top {N_TOP}": top}
        ),
        f"One-way ANOVA of each LR pair's scores, divided by the condition's "
        f"read depth, over "
        f"'{OUTCOME}': F, p, Benjamini-Hochberg-adjusted p across LR pairs, and "
        f"eta-squared (histogram). The whole component's eta-squared is "
        f"{result['component_eta']:.4f}.",
        part=0,
    )
    top_names = names[result["top"]]
    labels = shared_gene_labels(result["top"], lr_pair_genes(names))
    merged = np.array([" + ".join(top_names[labels == g]) for g in np.unique(labels)])
    pairs = np.triu_indices(len(merged), 1)
    common.source_data(
        "top_pair_rank_correlations",
        pd.DataFrame(
            {
                "LR pair 1": merged[pairs[0]],
                "LR pair 2": merged[pairs[1]],
                "Spearman correlation": result["observed"],
            }
        ),
        f"Spearman correlation across conditions of every pair of the component's "
        f"top {N_TOP} LR pairs, those sharing a ligand or receptor gene averaged "
        "into one (joined by '+'; their per-condition ranks averaged, then "
        "re-ranked).",
        part=1,
    )
    common.source_data(
        "baseline_rank_correlations",
        pd.concat(
            pd.DataFrame(draws)
            .rename_axis(index="Draw", columns="Pair of pairs")
            .stack()
            .dropna()
            .rename("Spearman correlation")
            .reset_index()
            .assign(Baseline=BASELINES[name][0])
            for name, draws in result["baselines"].items()
        ),
        f"The same correlations for {COVARIATION_DRAWS} draws (seeded) of "
        f"{N_TOP} LR pairs from outside the top pairs, per baseline: random, "
        f"expression-matched (one per top pair from its {N_EXPRESSION_NEIGHBORS} "
        "nearest pairs in log mean score), and other outcome-associated pairs "
        "(adjusted p < 0.05), each draw's pairs sharing a gene averaged into one "
        "as above.",
        part=1,
    )
    common.source_data(
        "covariation_tests",
        result["tests"],
        "The top pairs' mean correlation against each baseline's draws: "
        "one-sided empirical p, the share of draws (+1 smoothed, of "
        f"{COVARIATION_DRAWS} + 1) whose mean correlation is at least the top "
        "pairs', and Benjamini-Hochberg-adjusted p across baselines (stars).",
        part=1,
    )


def makeFigure():
    return common.standalone_fixed(__name__)
