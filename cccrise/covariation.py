"""Whether a component's LR pairs move together across conditions (Figure
3F, generalized to every method in Figure 5E).

Each LR pair is scored per condition in the component's own sender/receiver
context (`context_scores`: the communication tensor's slice for that pair,
weighted by the component's sender and receiver loadings and summed). Its
most heavily weighted pairs (`N_TOP` in Figure 3; in Figure 5 the fewest
carrying `CARRIED_SHARE` of its squared loading, `carrying_pairs`) are compared by their Spearman
correlation across conditions (`pairwise_rank_correlations`), against the
same correlations of LR pairs drawn from outside them (`covariation_baselines`).
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from .ccc_rise import calculate_interaction_tensor
from .factors import load_factors_netcdf
from .import_data import import_ligand_receptor_pairs
from .utils import load_rise_reference

#: The component's most heavily weighted LR pairs tested for co-variation
#: (Figure 3); at most this many in Figure 5 (`carrying_pairs`).
N_TOP = 10
#: Figure 5 tests the fewest top pairs carrying this share of a component's
#: squared LR loading (`carrying_pairs`).
CARRIED_SHARE = 0.9
COVARIATION_DRAWS = 2000
#: An expression-matched pair is one of a top pair's this many nearest.
N_EXPRESSION_NEIGHBORS = 20


def cccrise_tensor(
    factors_path: Path, adata: anndata.AnnData, condition_column: str
) -> np.ndarray:
    """cccRISE's raw tensor (condition x sender eigenstate x receiver
    eigenstate x LR pair, in the factors file's LR order), rebuilt from the
    saved RISE fit with the scoring the CP was fit on, which reproduces the
    fit's R2X."""
    factors = load_factors_netcdf(factors_path)
    adata = load_rise_reference(factors_path, adata, condition_column)
    tensor, lr_pairs = calculate_interaction_tensor(
        adata, import_ligand_receptor_pairs(), complex_sep=factors.attrs["complex_sep"]
    )
    order = pd.Index(lr_pairs["interaction_symbol"].astype(str)).get_indexer(
        factors["lr_pair"].values.astype(str)
    )
    return np.asarray(tensor)[..., order]


def context_scores(
    tensor: np.ndarray, sender: np.ndarray, receiver: np.ndarray
) -> np.ndarray:
    """Each LR pair's per-condition score in a component's sender/receiver
    context: the (condition, sender, receiver, LR) `tensor` weighted by the
    component's `sender` and `receiver` loadings and summed (conditions x LR
    pairs)."""
    return np.einsum("cijl,i,j->cl", tensor, sender, receiver)


def top_pairs(lr_loadings: np.ndarray, n: int = N_TOP) -> np.ndarray:
    """The indices of the `n` largest |LR loadings|, heaviest first."""
    return np.argsort(-np.abs(lr_loadings))[:n]


def carrying_pairs(
    lr_loadings: np.ndarray, share: float = CARRIED_SHARE, cap: int = N_TOP
) -> np.ndarray:
    """`top_pairs`, as many as the fewest that carry `share` of the squared
    loading, at most `cap`."""
    squared = np.sort(lr_loadings**2)[::-1]
    carried = np.cumsum(squared) / squared.sum()
    n = int(np.searchsorted(carried, share - 1e-12)) + 1
    return top_pairs(lr_loadings, min(n, cap))


def pairwise_rank_correlations(scores: np.ndarray, columns, genes=None) -> np.ndarray:
    """Spearman correlation across conditions of every pair of `columns`
    (upper triangle): each LR pair's scores are ranked across conditions and
    the Pearson correlation of those ranks taken. With `genes` (each LR
    pair's set of genes, every subunit), pairs of pairs that share a gene,
    and so co-vary almost by construction, are NaN."""
    columns = np.asarray(columns)
    ranks = np.argsort(np.argsort(scores[:, columns], axis=0), axis=0)
    upper = np.triu_indices(len(columns), 1)
    correlations = np.corrcoef(ranks.T)[upper]
    if genes is not None:
        shared = [bool(genes[columns[i]] & genes[columns[j]]) for i, j in zip(*upper)]
        correlations[np.array(shared, bool)] = np.nan
    return correlations


def covariation_baselines(
    scores: np.ndarray,
    top: np.ndarray,
    associated: np.ndarray | None = None,
    seed: int = 0,
    genes=None,
) -> dict[str, np.ndarray]:
    """`pairwise_rank_correlations` (draws x pairs of pairs) of as many LR
    pairs as `top`, drawn `COVARIATION_DRAWS` times from the LR pairs outside
    `top`, under each baseline:

    - random: drawn uniformly without replacement.
    - expression_matched: one pair per top pair, each drawn from that top
      pair's `N_EXPRESSION_NEIGHBORS` nearest pairs in log mean score
      (without replacement within a draw), so the draw matches the top pairs'
      expression levels.
    - associated (when `associated`, a boolean mask, is given): drawn
      uniformly from the other pairs it marks, i.e. those significantly
      associated with the outcome; all of them when there are fewer than
      `top`.

    With `genes`, correlations between pairs sharing a gene are NaN
    (`pairwise_rank_correlations`); the draws themselves are unchanged."""
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

    draws = {
        "random": lambda: rng.choice(pool, len(top), replace=False),
        "expression_matched": matched_draw,
    }
    if associated is not None:
        associated_pool = np.setdiff1d(np.flatnonzero(associated), top)
        draws["associated"] = lambda: rng.choice(
            associated_pool, min(len(top), associated_pool.size), replace=False
        )
    return {
        name: np.array(
            [
                pairwise_rank_correlations(scores, draw(), genes)
                for _ in range(COVARIATION_DRAWS)
            ]
        )
        for name, draw in draws.items()
    }
