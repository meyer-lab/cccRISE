"""Coordinated changes: the outcome-associated CP component against its LR
pairs scored on their own, from the raw communication tensor.

Every LR pair is scored per condition in the component's own sender/receiver
context (`component_lr_scores`: the raw tensor slice for that pair, weighted
by the component's sender and receiver eigenstate loadings and summed over
eigenstates).

- Association: each pair's scores, and the component's condition loadings,
  get the component ANOVA against the 6-month outcome
  (`column_associations`: one-way ANOVA over the outcome groups, eta^2 as the
  effect size, Benjamini-Hochberg FDR across the 513 LR pairs). The histogram
  is every LR pair's eta^2, with the whole component's eta^2 and its `N_TOP`
  most heavily weighted LR pairs' marked.
- Co-variation of the `N_TOP` pairs: the Spearman correlation across
  conditions of every pair of them (`pairwise_rank_correlations`), ranks so a
  few conditions in which many LR pairs spike together do not dominate it.
  Each baseline box pools those correlations over `COVARIATION_DRAWS` draws of
  as many LR pairs from outside that set (`covariation_baselines`): random,
  expression-matched (one pair per top pair from its `N_EXPRESSION_NEIGHBORS`
  nearest pairs in log mean score), and other associated (FDR < 0.05 on their
  own), so every box is a distribution of single pairwise correlations.
"""

import copy
from functools import cache

import anndata
import numpy as np
import pandas as pd

from ...anova import run_multiple_factors_anova
from ...covariation import (
    COVARIATION_DRAWS,
    N_EXPRESSION_NEIGHBORS,
    N_TOP,
    cccrise_tensor,
    context_scores,
    covariation_baselines,
    pairwise_rank_correlations,
    top_pairs,
)
from ...factors import load_factors_netcdf
from ...runners.datasets import DATASETS
from .. import common, style
from ._common import (
    COMPONENT,
    DATASET,
    OUTCOME,
    SPEC,
    association,
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


def plot(axs):
    result = coordinated_analysis()
    _source_data(result)
    plot_lr_association_distribution(
        result["lr_eta"], result["component_eta"], result["top"], axs[0]
    )
    plot_lr_covariation(result["observed"], result["baselines"], axs[1])


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
        borderaxespad=0,
    ).set_zorder(5)


def plot_lr_covariation(
    observed: np.ndarray, baselines: dict[str, np.ndarray], ax
) -> None:
    """Pairwise rank correlations of the top pairs (boxed, with each pair of
    pairs as a point) beside every draw's pairwise correlations, pooled, per
    baseline."""
    groups = {f"Cmp. {COMPONENT} top {N_TOP}": observed} | {
        BASELINES[name][0]: draws.ravel() for name, draws in baselines.items()
    }
    # Green, as the top pairs are in the association panel
    colors = [style.COLORS["highlight"]] + [color for _, color in BASELINES.values()]
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
    common.reference_line(ax, 0)
    ax.set_xticks(
        range(1, len(groups) + 1),
        list(groups),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Pairwise rank correlation\nacross conditions")


@cache
def coordinated_analysis() -> dict:
    """The component's and every LR pair's outcome association, the
    component's top LR pairs, and those pairs' co-variation against
    `covariation_baselines`."""
    adata = factors_adata()
    scores = component_lr_scores(DATASET, COMPONENT)
    top = top_pairs(np.asarray(adata.uns["D"])[:, COMPONENT - 1])
    per_lr = column_associations(scores, adata)
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
        "observed": pairwise_rank_correlations(scores, top),
        "baselines": covariation_baselines(
            scores, top, per_lr.PAdj.to_numpy() < common.SIG_ALPHA
        ),
    }


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
        f"Each LR pair's raw communication score per condition in CP component "
        f"{COMPONENT}'s sender/receiver context (the tensor slice weighted by "
        "the component's sender and receiver eigenstate loadings, summed).",
    )
    top = np.isin(np.arange(len(names)), result["top"])
    common.source_data(
        "lr_pair_outcome_anova",
        result["per_lr"][["F", "P", "PAdj", "EtaSquared"]].assign(
            **{"LR pair": names, f"Cmp {COMPONENT} top {N_TOP}": top}
        ),
        f"One-way ANOVA of each LR pair's read-depth-corrected scores over "
        f"'{OUTCOME}': F, p, Benjamini-Hochberg-adjusted p across LR pairs, and "
        f"eta-squared (histogram). The whole component's eta-squared is "
        f"{result['component_eta']:.4f}.",
        part=0,
    )
    pairs = np.triu_indices(N_TOP, 1)
    top_names = names[result["top"]]
    common.source_data(
        "top_pair_rank_correlations",
        pd.DataFrame(
            {
                "LR pair 1": top_names[pairs[0]],
                "LR pair 2": top_names[pairs[1]],
                "Spearman correlation": result["observed"],
            }
        ),
        f"Spearman correlation across conditions of every pair of the component's "
        f"top {N_TOP} LR pairs.",
        part=1,
    )
    common.source_data(
        "baseline_rank_correlations",
        pd.concat(
            pd.DataFrame(draws)
            .rename_axis(index="Draw", columns="Pair of pairs")
            .stack()
            .rename("Spearman correlation")
            .reset_index()
            .assign(Baseline=BASELINES[name][0])
            for name, draws in result["baselines"].items()
        ),
        f"The same correlations for {COVARIATION_DRAWS} draws (seeded) of "
        f"{N_TOP} LR pairs from outside the top pairs, per baseline: random, "
        f"expression-matched (one per top pair from its {N_EXPRESSION_NEIGHBORS} "
        "nearest pairs in log mean score), and other outcome-associated pairs "
        "(adjusted p < 0.05).",
        part=1,
    )


def component_lr_scores(dataset: str, component: int) -> np.ndarray:
    """Each LR pair's per-condition raw communication score in a CP
    component's own sender/receiver context (conditions x LR pairs, in the
    factors file's LR order): each LR pair's slice of `communication_tensor`
    weighted by the component's sender (B) and receiver (C) eigenstate
    loadings and summed over eigenstates, so a pair is scored between the
    same cells the component describes."""
    factors = load_factors_netcdf(DATASETS[dataset].factors_path())
    sender, receiver = (
        factors[k].values[:, component - 1].astype(float) for k in ("cp_B", "cp_C")
    )
    return context_scores(communication_tensor(dataset), sender, receiver)


@cache
def communication_tensor(dataset: str) -> np.ndarray:
    """The raw tensor (condition x sender eigenstate x receiver eigenstate x
    LR pair, in the factors file's LR order), rebuilt from the saved RISE fit
    with the scoring the CP was fit on, which reproduces the fit's R2X."""
    spec = DATASETS[dataset]
    return cccrise_tensor(spec.factors_path(), spec.load(), spec.condition_column)


def column_associations(matrix: np.ndarray, adata: anndata.AnnData) -> pd.DataFrame:
    """The component ANOVA against OUTCOME, applied to each column of a
    conditions x features `matrix` in place of the component condition
    loadings: a one-way ANOVA per column over the association's groups, each
    condition's row read-depth corrected as the loadings are, reporting
    eta^2, FDR-corrected (Benjamini-Hochberg) across columns."""
    adata = copy.copy(adata)
    adata.uns = {**adata.uns, "A": matrix}
    return run_multiple_factors_anova(
        adata,
        [association(OUTCOME)],
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
    )


def makeFigure():
    return common.standalone_fixed(__name__)
