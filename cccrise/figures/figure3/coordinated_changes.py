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
from ...ccc_rise import calculate_interaction_tensor
from ...factors import load_factors_netcdf
from ...import_data import import_ligand_receptor_pairs
from ...runners.datasets import DATASETS
from ...utils import load_rise_reference
from .. import common, style
from ._common import COMPONENT, DATASET, OUTCOME, SPEC, association, factors_adata

# The component's N_TOP most heavily weighted LR pairs are the ones tested
# for co-variation.
N_TOP = 10
COVARIATION_DRAWS = 2000
N_EXPRESSION_NEIGHBORS = 20  # an expression-matched pair is one of these nearest
BASELINES = {  # label, color
    "random": ("Random", style.COLORS["muted"]),
    "expression_matched": ("Expression-matched", style.COLORS["receiver"]),
    "associated": ("Other associated", style.COLORS["up"]),
}
AXES = ("square", "square")


def plot(axs):
    result = coordinated_analysis()
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
    weights = np.asarray(adata.uns["D"])[:, COMPONENT - 1]
    top = np.argsort(-np.abs(weights))[:N_TOP]
    per_lr = column_associations(scores, adata)
    loading = np.asarray(adata.uns["A"])[:, [COMPONENT - 1]]
    return {
        "lr_eta": per_lr.EtaSquared.to_numpy(),
        "component_eta": float(column_associations(loading, adata).EtaSquared.iloc[0]),
        "top": top,
        "observed": pairwise_rank_correlations(scores, top),
        "baselines": covariation_baselines(
            scores, top, per_lr.PAdj.to_numpy() < common.SIG_ALPHA
        ),
    }


def component_lr_scores(dataset: str, component: int) -> np.ndarray:
    """Each LR pair's per-condition raw communication score in a CP
    component's own sender/receiver context (conditions x LR pairs, in the
    factors file's LR order).

    The raw tensor (condition x sender eigenstate x receiver eigenstate x LR
    pair) is rebuilt from the saved RISE fit with the scoring the CP was fit
    on, which reproduces the fit's R2X; each LR pair's slice is then weighted
    by the component's sender (B) and receiver (C) eigenstate loadings and
    summed over eigenstates, so a pair is scored between the same cells the
    component describes."""
    spec = DATASETS[dataset]
    factors = load_factors_netcdf(spec.factors_path())
    adata = load_rise_reference(spec.factors_path(), spec.load(), spec.condition_column)
    tensor, lr_pairs = calculate_interaction_tensor(
        adata, import_ligand_receptor_pairs(), complex_sep=factors.attrs["complex_sep"]
    )
    order = pd.Index(lr_pairs["interaction_symbol"].astype(str)).get_indexer(
        factors["lr_pair"].values.astype(str)
    )
    sender, receiver = (
        factors[k].values[:, component - 1].astype(float) for k in ("cp_B", "cp_C")
    )
    return np.einsum("cijl,i,j->cl", np.asarray(tensor)[..., order], sender, receiver)


def column_associations(matrix: np.ndarray, adata: anndata.AnnData) -> pd.DataFrame:
    """The component ANOVA against OUTCOME, applied to each column of a
    conditions x features `matrix` in place of the component condition
    loadings: a one-way ANOVA per column over the association's groups,
    reporting eta^2, FDR-corrected (Benjamini-Hochberg) across columns."""
    adata = copy.copy(adata)
    adata.uns = {**adata.uns, "A": matrix}
    return run_multiple_factors_anova(
        adata,
        [association(OUTCOME)],
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
    )


def pairwise_rank_correlations(scores: np.ndarray, columns) -> np.ndarray:
    """Spearman correlation across conditions of every pair of `columns`
    (upper triangle): each LR pair's scores are ranked across conditions and
    the Pearson correlation of those ranks taken."""
    ranks = np.argsort(np.argsort(scores[:, columns], axis=0), axis=0)
    return np.corrcoef(ranks.T)[np.triu_indices(len(columns), 1)]


def covariation_baselines(
    scores: np.ndarray, top: np.ndarray, associated: np.ndarray, seed: int = 0
) -> dict[str, np.ndarray]:
    """`pairwise_rank_correlations` (draws x pairs of pairs) of as many LR
    pairs as `top`, drawn `COVARIATION_DRAWS` times from the LR pairs outside
    `top`, under three baselines:

    - random: drawn uniformly without replacement.
    - expression_matched: one pair per top pair, each drawn from that top
      pair's `N_EXPRESSION_NEIGHBORS` nearest pairs in log mean score
      (without replacement within a draw), so the draw matches the top pairs'
      expression levels.
    - associated: drawn uniformly from the `associated` (boolean mask) pairs,
      i.e. the other pairs significantly associated with the outcome."""
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(scores.shape[1]), top)
    level = np.log10(scores.mean(axis=0) + 1e-12)
    neighbors = [
        pool[np.argsort(np.abs(level[pool] - level[t]))[:N_EXPRESSION_NEIGHBORS]]
        for t in top
    ]

    def matched_draw():
        draw = []
        for candidates in neighbors:
            draw.append(rng.choice(np.setdiff1d(candidates, draw)))
        return draw

    associated_pool = np.setdiff1d(np.flatnonzero(associated), top)
    draws = {
        "random": lambda: rng.choice(pool, len(top), replace=False),
        "expression_matched": matched_draw,
        "associated": lambda: rng.choice(associated_pool, len(top), replace=False),
    }
    return {
        name: np.array(
            [
                pairwise_rank_correlations(scores, draw())
                for _ in range(COVARIATION_DRAWS)
            ]
        )
        for name, draw in draws.items()
    }


def makeFigure():
    return common.standalone_fixed(__name__)
