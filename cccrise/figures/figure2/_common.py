"""Choices and data shared by Figure 2's panels.

Cell type granularity (`cmp_eigenstate_loadings`, `cmp_weight_scatter`,
`eigenstate_scatter`): the CP component that splits a cell type into sender
and receiver subpopulations, and the RISE eigenstates its sender and receiver
loadings draw on.

Coordinated changes (`lr_association`, `lr_covariation`): the
outcome-associated CP component's LR pairs, each scored on its own in the raw
communication tensor and compared to the component.
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
from .. import common
from ..figure3._common import SPEC, association, factors_adata

DATASET = "alad"

## Cell type granularity
GRANULARITY_COMPONENT = 5
# Its largest sender (B) and receiver (C) loadings
SENDER_EIGENSTATE, RECEIVER_EIGENSTATE = 17, 15
HIGHLIGHT_CELLTYPE = "Epithelial cells"

## Coordinated changes
# The CP component associated with ASSOCIATION (Figure 3); its N_TOP most
# heavily weighted LR pairs are the ones tested for co-variation.
ASSOCIATED_COMPONENT = 4
ASSOCIATION = "6-Month Outcome (4-Way)"
N_TOP = 10
COVARIATION_DRAWS = 2000
N_EXPRESSION_NEIGHBORS = 20  # an expression-matched pair is one of these nearest
# Beside the axes: the panels are too short for an in-axes legend corner
OUTSIDE_LEGEND = {"loc": "center left", "bbox_to_anchor": (1.02, 0.5)}


@cache
def coordinated_analysis() -> dict:
    """The component's and every LR pair's outcome association, the
    component's top LR pairs, and those pairs' co-variation against
    `covariation_baselines`."""
    adata = factors_adata()
    scores = component_lr_scores(DATASET, ASSOCIATED_COMPONENT)
    weights = np.asarray(adata.uns["D"])[:, ASSOCIATED_COMPONENT - 1]
    top = np.argsort(-np.abs(weights))[:N_TOP]
    per_lr = column_associations(scores, adata)
    loading = np.asarray(adata.uns["A"])[:, [ASSOCIATED_COMPONENT - 1]]
    return {
        "lr_eta": per_lr.EtaSquared.to_numpy(),
        "component_eta": float(column_associations(loading, adata).EtaSquared.iloc[0]),
        "top": top,
        "observed": mean_pairwise_rank_correlation(scores, top),
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
    """Figure 3's component ANOVA against ASSOCIATION, applied to each column
    of a conditions x features `matrix` in place of the component condition
    loadings: a one-way ANOVA per column over the association's groups,
    reporting eta^2, FDR-corrected (Benjamini-Hochberg) across columns."""
    adata = copy.copy(adata)
    adata.uns = {**adata.uns, "A": matrix}
    return run_multiple_factors_anova(
        adata,
        [association(ASSOCIATION)],
        condition_column=SPEC.condition_column,
        correction="fdr_bh",
    )


def mean_pairwise_rank_correlation(scores: np.ndarray, columns) -> float:
    """Mean Spearman correlation across conditions over every pair of
    `columns`: each LR pair's scores are ranked across conditions, and the
    Pearson correlation of those ranks is averaged over all pairs."""
    ranks = np.argsort(np.argsort(scores[:, columns], axis=0), axis=0)
    correlations = np.corrcoef(ranks.T)
    return float(correlations[np.triu_indices(len(columns), 1)].mean())


def covariation_baselines(
    scores: np.ndarray, top: np.ndarray, associated: np.ndarray, seed: int = 0
) -> dict[str, np.ndarray]:
    """`mean_pairwise_rank_correlation` of as many LR pairs as `top`, drawn
    `COVARIATION_DRAWS` times from the LR pairs outside `top`, under three
    baselines:

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
                mean_pairwise_rank_correlation(scores, draw())
                for _ in range(COVARIATION_DRAWS)
            ]
        )
        for name, draw in draws.items()
    }
