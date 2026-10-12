"""What one cell-type labelling lets each comparator recover, for the
granularity sampling (`run_granularity_sampling`; labellings from
`run_comparator_fits`).

Per labelling, for every own-rank comparator fit against the published
cccRISE fit:

- per cccRISE component: its LR column's containment in the comparator's LR
  span and its condition column's containment in the comparator's condition
  span (`agreement.per_column_containment`);
- per comparator column: its LR containment in cccRISE's LR span, its best
  cccRISE match by the per-pair min weight on the shared top-`k` pairs
  (`lr_concordance.shared_min_weight`) and whether that reaches `LR_MATCH`,
  and its effective number of LR pairs (1 / sum of squared loading shares
  squared; below `SINGLE_PAIR` it is counted single-pair)."""

import numpy as np
import pandas as pd

from .agreement import aligned_modes, per_column_containment
from .lr_concordance import PRIMARY_K, shared_min_weight, top_k_sets

LR_MATCH = 0.5
SINGLE_PAIR = 1.5


def effective_pairs(lr: np.ndarray) -> np.ndarray:
    """Per column: 1 / sum of its squared-loading shares squared."""
    sq = np.asarray(lr, float) ** 2
    share = sq / sq.sum(axis=0)
    return 1 / (share**2).sum(axis=0)


def _component_rows(ref, fit, modes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cccrise_component": np.arange(1, ref.condition_factor.shape[1] + 1),
            "lr_containment": per_column_containment(modes["lr"], ref.modes["lr"]),
            "condition_containment": per_column_containment(
                modes["condition"], ref.condition_factor
            ),
        }
    )


def _column_rows(ref, modes, k) -> pd.DataFrame:
    lr, ref_lr = modes["lr"], ref.modes["lr"]
    weight = shared_min_weight(top_k_sets(ref_lr, k), top_k_sets(lr, k), ref_lr, lr)
    best = weight.argmax(axis=0)
    eff = effective_pairs(lr)
    return pd.DataFrame(
        {
            "column": np.arange(1, lr.shape[1] + 1),
            "reverse_containment": per_column_containment(ref_lr, lr),
            "best_component": best + 1,
            "min_weight": weight[best, np.arange(lr.shape[1])],
            "lr_match": weight[best, np.arange(lr.shape[1])] >= LR_MATCH,
            "effective_pairs": eff,
            "single_pair": eff < SINGLE_PAIR,
        }
    )


def labelling_metrics(fits, info, lr_names, k: int = PRIMARY_K):
    """The component table and the column table of one labelling's fits
    (`benchmark.read_fits`; the cccRISE fit is the reference)."""
    ref = next(f for f in fits if f.method == "cccRISE")
    components, columns = [], []
    for fit in (f for f in fits if f.method != "cccRISE" and f.rank_mode == "own"):
        modes = aligned_modes(fit, info, lr_names)
        label = {"method": fit.method, "rank": str(fit.rank)}
        components.append(_component_rows(ref, fit, modes).assign(**label))
        columns.append(_column_rows(ref, modes, k).assign(**label))
    return pd.concat(components, ignore_index=True), pd.concat(
        columns, ignore_index=True
    )
