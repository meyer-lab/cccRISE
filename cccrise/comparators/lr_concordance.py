"""Do cccRISE and the tensor comparators agree on which LR pairs make a program?

Readers of a communication program look at its top-loaded ligand-receptor
pairs, so agreement is measured on those rather than on whole columns:

- `top_k_sets` / `shared_counts`: how many of two LR columns' top-k pairs
  (by |loading|) coincide, for several k, so that agreement in the head can be
  told apart from agreement deeper in the list (`head_spearman`).
- `abundance_null`: the shared count two columns would reach if each simply
  drew its top-k from the abundant pairs (draws weighted by each pair's mean
  communication score). Every method favours abundant pairs, so a uniform
  null would overstate agreement.
- `coverage_table`: whether one cccRISE component's top-k is spread over
  several columns of one comparator fit.
- `missed_pairs`: pairs in a comparator's top-k that no cccRISE component
  ranks highly, and the reverse.
- `classify_components`: every CP comparator component, classified once as
  recovered by cccRISE, same LR program on different cells or samples, or not
  recovered (one-sample or broad).

Comparator LR columns put nearly all their weight on one or two pairs, so a
rank-only overlap can be carried by pairs with negligible loading;
`shared_mass` / `head_mass` report how much weight the shared pairs carry.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

from .agreement import unit_columns

KS = (5, 10, 25)
#: k at which hits are called for matching and classification.
PRIMARY_K = 10
#: A pair ranked below this in every component of a fit is "missed" by it.
MISSED_RANK = 50
#: |cos| at or above this counts as agreement on a mode (`classify_components`).
AGREE = 0.5
#: Post-hoc mass-aware hit: the shared top-k pairs carry at least this share
#: of the comparator column's squared loading (added after the LR columns
#: turned out to put >= 94% of their mass on 1-2 pairs).
MASS_HIT = 0.5
#: Condition participation ratio below this marks a one-sample component.
ONE_SAMPLE = 1.5


def top_k_sets(lr: np.ndarray, k: int) -> list[frozenset[int]]:
    """Per column of `lr` (pairs x components): its `k` largest |loadings|."""
    order = np.argsort(-np.abs(np.asarray(lr, float)), axis=0, kind="stable")[:k]
    return [frozenset(order[:, c].tolist()) for c in range(order.shape[1])]


def shared_counts(a: Sequence[frozenset], b: Sequence[frozenset]) -> np.ndarray:
    """``(len(a), len(b))`` number of pairs every two top-k sets share."""
    return np.array([[len(x & y) for y in b] for x in a], dtype=int)


def shared_mass(
    a_sets: Sequence[frozenset], b_sets: Sequence[frozenset], b: np.ndarray
) -> np.ndarray:
    """``(len(a_sets), b cols)``: the share of each `b` column's squared
    loading on the pairs its top-k shares with each `a` top-k set."""
    sq = np.asarray(b, float) ** 2
    total = sq.sum(axis=0)
    return np.array(
        [
            [sq[sorted(x & y), j].sum() / total[j] for j, y in enumerate(b_sets)]
            for x in a_sets
        ]
    )


def head_mass(lr: np.ndarray, ks: Sequence[int]) -> pd.DataFrame:
    """Per column and k: the share of its squared loading on its top-k pairs."""
    sq = np.sort(np.asarray(lr, float) ** 2, axis=0)[::-1]
    cum = np.cumsum(sq, axis=0) / sq.sum(axis=0)
    return pd.DataFrame(
        [
            {"component": c + 1, "k": k, "head_mass": float(cum[k - 1, c])}
            for k in ks
            for c in range(lr.shape[1])
        ]
    )


def loading_profile(lr: np.ndarray, names: Sequence[str], depth: int) -> pd.DataFrame:
    """Per column, its `depth` largest pairs by |loading|: the pair, its
    |loading| relative to the column's largest, its share of the column's
    squared loading and the cumulative share down to its rank."""
    x = np.abs(np.asarray(lr, float))
    order = np.argsort(-x, axis=0, kind="stable")[:depth]
    top = np.take_along_axis(x, order, axis=0)
    share = top**2 / (x**2).sum(axis=0)
    return pd.DataFrame(
        {
            "component": np.tile(np.arange(1, x.shape[1] + 1), len(order)),
            "lr_rank": np.repeat(np.arange(1, len(order) + 1), x.shape[1]),
            "lr_pair": np.asarray(names).astype(str)[order].ravel(),
            "rel_max": (top / top[0]).ravel(),
            "share": share.ravel(),
            "cum_share": np.cumsum(share, axis=0).ravel(),
        }
    )


def ranks(lr: np.ndarray) -> np.ndarray:
    """Per column, each pair's rank by |loading| (1 = largest)."""
    order = np.argsort(-np.abs(np.asarray(lr, float)), axis=0, kind="stable")
    out = np.empty_like(order)
    rows = np.arange(1, order.shape[0] + 1)
    for c in range(order.shape[1]):
        out[order[:, c], c] = rows
    return out


def _weighted_sets(
    weights: np.ndarray, k: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_draws, k)`` indices drawn without replacement with probability
    proportional to `weights` (Gumbel top-k)."""
    with np.errstate(divide="ignore"):
        logw = np.log(np.asarray(weights, float))
    keys = logw[None, :] + rng.gumbel(size=(n_draws, len(logw)))
    return np.argpartition(-keys, k - 1, axis=1)[:, :k]


def abundance_null(
    weights: np.ndarray, k: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Shared counts of two independent abundance-weighted top-k draws."""
    n = len(weights)
    a = np.zeros((n_draws, n), dtype=bool)
    b = np.zeros((n_draws, n), dtype=bool)
    rows = np.arange(n_draws)[:, None]
    a[rows, _weighted_sets(weights, k, n_draws, rng)] = True
    b[rows, _weighted_sets(weights, k, n_draws, rng)] = True
    return (a & b).sum(axis=1)


def null_thresholds(
    weights: np.ndarray, ks: Sequence[int], n_draws: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Per k: the abundance-weighted and uniform (hypergeometric) nulls' mean
    and 95th percentile of the shared count."""
    n = len(weights)
    rows = []
    for k in ks:
        null = abundance_null(weights, k, n_draws, rng)
        rows.append(
            {
                "k": k,
                "null_mean": float(null.mean()),
                "null_q95": float(np.quantile(null, 0.95)),
                "uniform_mean": k * k / n,
                "uniform_q95": float(hypergeom.ppf(0.95, n, k, k)),
            }
        )
    return pd.DataFrame(rows)


def overlap_table(
    ref: np.ndarray, other: np.ndarray, ks: Sequence[int], thresholds: pd.DataFrame
) -> pd.DataFrame:
    """Every (ref component, other column, k): shared count, overlap fraction,
    and whether it is a hit (above the abundance null's 95th percentile)."""
    q95 = thresholds.set_index("k")["null_q95"]
    cos = np.abs(unit_columns(ref).T @ unit_columns(other))
    frames = []
    for k in ks:
        ref_sets, other_sets = top_k_sets(ref, k), top_k_sets(other, k)
        counts = shared_counts(ref_sets, other_sets)
        mass_other = shared_mass(ref_sets, other_sets, other)
        mass_ref = shared_mass(other_sets, ref_sets, ref).T
        i, j = np.indices(counts.shape)
        frames.append(
            pd.DataFrame(
                {
                    "cccrise_component": i.ravel() + 1,
                    "comparator_component": j.ravel() + 1,
                    "k": k,
                    "shared": counts.ravel(),
                    "overlap": counts.ravel() / k,
                    "hit": counts.ravel() > q95[k],
                    "mass_comparator": mass_other.ravel(),
                    "mass_cccrise": mass_ref.ravel(),
                    "lr_abs_cos": cos.ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def best_matches(overlaps: pd.DataFrame, k: int = PRIMARY_K) -> pd.DataFrame:
    """Per comparator column: the cccRISE component sharing most of its top-k
    pairs, ties broken by |cos| of the LR columns."""
    sub = overlaps[overlaps["k"] == k].sort_values(
        ["comparator_component", "shared", "lr_abs_cos"],
        ascending=[True, False, False],
    )
    return sub.groupby("comparator_component", as_index=False).head(1)


def head_spearman(a: np.ndarray, b: np.ndarray, k: int = max(KS)) -> float:
    """Spearman correlation of two LR columns over the union of their top-k."""
    union = sorted(top_k_sets(a[:, None], k)[0] | top_k_sets(b[:, None], k)[0])
    return float(spearmanr(np.abs(a[union]), np.abs(b[union])).statistic)


def coverage_table(
    ref: np.ndarray, other: np.ndarray, ks: Sequence[int]
) -> pd.DataFrame:
    """Per k and direction: the fraction of each column's top-k found in the
    union of the other fit's top-k, next to its best single-column overlap."""
    rows = []
    for k in ks:
        sets = {"cccRISE": top_k_sets(ref, k), "comparator": top_k_sets(other, k)}
        for side, against in (("cccRISE", "comparator"), ("comparator", "cccRISE")):
            union = frozenset().union(*sets[against])
            for c, s in enumerate(sets[side], start=1):
                rows.append(
                    {
                        "direction": f"{side} in {against}",
                        "component": c,
                        "k": k,
                        "coverage": len(s & union) / k,
                        "best_single": max(len(s & t) for t in sets[against]) / k,
                    }
                )
    return pd.DataFrame(rows)


def whole_model_percentile(lr: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Per pair: the percentile of its weight over a CP model's components,
    sum_c lambda_c |D[p, c]| (1 = the most weighted pair)."""
    w = np.ones(lr.shape[1]) if weights is None else np.abs(np.asarray(weights, float))
    score = np.abs(np.asarray(lr, float)) @ w
    return pd.Series(score).rank(pct=True).to_numpy()


def missed_pairs(
    source: np.ndarray,
    target: np.ndarray,
    target_weights: np.ndarray | None,
    k: int,
    names: Sequence[str],
) -> pd.DataFrame:
    """Every pair in any `source` column's top-k: the source columns holding
    it, its largest share of a source column's squared loading (whether it
    is a column's head or its near-zero tail), its best rank in any `target` column, its whole-model percentile in
    `target`, and whether `target` misses it (rank > `MISSED_RANK` everywhere)."""
    names = np.asarray(names).astype(str)
    sets = top_k_sets(source, k)
    best = ranks(target).min(axis=1)
    sq = np.asarray(source, float) ** 2
    share = sq / sq.sum(axis=0)
    pct = whole_model_percentile(target, target_weights)
    rows = []
    for p in sorted(frozenset().union(*sets)):
        rows.append(
            {
                "lr_pair": names[p],
                "k": k,
                "source_components": ",".join(
                    str(c + 1) for c, s in enumerate(sets) if p in s
                ),
                "source_max_share": float(
                    max(share[p, c] for c, s in enumerate(sets) if p in s)
                ),
                "best_target_rank": int(best[p]),
                "target_percentile": float(pct[p]),
                "missed": bool(best[p] > MISSED_RANK),
            }
        )
    return pd.DataFrame(rows)


def mode_agreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a cols, b cols)`` |cosine| between columns of two factors."""
    return np.abs(unit_columns(a).T @ unit_columns(b))


def classify_components(matches: pd.DataFrame, hit: str = "hit") -> pd.Series:
    """A5 class of each comparator column, from its best match's `hit` column
    and its ``sender_cos`` / ``receiver_cos`` / ``condition_cos`` and
    ``condition_pr`` (participation ratio) columns."""
    agree = (
        matches[["sender_cos", "receiver_cos", "condition_cos"]].ge(AGREE).all(axis=1)
    )
    one_sample = matches["condition_pr"] < ONE_SAMPLE
    hits = matches[hit].astype(bool)
    classes = np.select(
        [hits & agree, hits, one_sample],
        ["recovered", "same LR, different who/where", "not recovered, one sample"],
        default="not recovered, broad",
    )
    return pd.Series(classes, index=matches.index, name="lr_class")
