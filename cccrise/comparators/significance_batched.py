"""Batched ``InteractionProgramSignificance`` on NumPy or CuPy.

`scriabin.module_significance` runs upstream's test one random LR set at a
time on the CPU. For a cohort (IBD: ~740 programs x 343 samples x 10,000
replicates) that is days, and R is ~15x slower still. Here the replicates
of one (sample, program) are processed as a batch, on the GPU when CuPy is
given:

- the random sets' similarity blocks are gathered at once
  (``m_cor[vars, vars]``, diagonal included, as upstream);
- each one-sided Wilcoxon p (`scriabin.wilcox_greater_p`: normal
  approximation with continuity and tie correction, as R's ``wilcox.test(...,
  exact = FALSE)``) is computed with the same integer-valued arithmetic, so
  W and the tie term are exact in float64; non-finite entries are dropped
  per row, as R drops NA;
- the value is the fraction of replicates with p > 0.05.

With the same random sets, the result equals `scriabin.module_significance`
(`rcheck/significance_check.py`, and against R with the parity gate's sets).
Random sets are drawn here as the first ``len(members)`` positions of a
random permutation (argsort of uniform keys), uniform without replacement
like upstream's ``sample``, from a per-(sample, program) seed.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

#: Elements (replicates x block entries) per batch; bounds GPU memory to a
#: few hundred MB of float64 per array.
BATCH_ELEMENTS = 40_000_000


def array_module(device: str) -> Any:
    """NumPy (``cpu``) or CuPy (``gpu``)."""
    if device == "gpu":
        import cupy

        return cupy
    return np


def _ndtr(xp: Any):
    if xp is np:
        return ndtr
    from cupyx.scipy.special import ndtr as cupy_ndtr

    return cupy_ndtr


def greater_p(a_sorted: Any, b: Any, xp: Any) -> Any:
    """``wilcox.test(a, b[i], alternative = "greater", exact = FALSE)$p.value``
    for every row of `b`, its non-finite entries dropped; `a_sorted` is
    finite and sorted. Same arithmetic as `scriabin.wilcox_greater_p`."""
    na = a_sorted.size
    rows, width_b = b.shape
    valid = xp.isfinite(b)
    nb = valid.sum(axis=1).astype(xp.float64)
    probe = xp.where(valid, b, xp.inf)
    below = xp.searchsorted(a_sorted, probe, side="left")
    upto = xp.searchsorted(a_sorted, probe, side="right")
    w = xp.where(valid, (na - upto) + 0.5 * (upto - below), 0.0).sum(axis=1)
    # Ties of the combined sample; dropped entries become NaN, each its own
    # run of length 1, which contributes t^3 - t = 0
    combined = xp.sort(
        xp.concatenate(
            [xp.broadcast_to(a_sorted, (rows, na)), xp.where(valid, b, xp.nan)], axis=1
        ),
        axis=1,
    )
    width = na + width_b
    starts = xp.ones(combined.shape, dtype=bool)
    starts[:, 1:] = combined[:, 1:] != combined[:, :-1]
    flat = xp.nonzero(starts.ravel())[0]
    runs = xp.diff(xp.concatenate([flat, xp.asarray([combined.size])])).astype(
        xp.float64
    )
    ties = xp.bincount(flat // width, weights=runs**3 - runs, minlength=rows)
    n = na + nb
    z = w - na * nb / 2
    sigma = xp.sqrt((na * nb / 12) * ((n + 1) - ties / (n * (n - 1))))
    return _ndtr(xp)(-(z - 0.5) / sigma)


def random_sets(n: int, k: int, replicates: int, seed: int, xp: Any) -> Any:
    """`replicates` uniform random `k`-subsets of ``range(n)`` (rows)."""
    keys = xp.random.RandomState(seed).random_sample((replicates, n))
    return xp.argsort(keys, axis=1)[:, :k]


def module_value(
    sim: Any,
    present: np.ndarray,
    n_members: int,
    sets: Any,
    xp: Any,
) -> float:
    """`scriabin.module_significance` for one program: `sim` the sample's
    similarity (device array), `present` the program's members' positions in
    it, `n_members` its size, `sets` the random sets (replicates x
    `n_members`). 1 where upstream's ``try`` would fail."""
    if len(present) == 0 or n_members > sim.shape[0]:
        return 1.0
    idx = xp.asarray(present)
    a = sim[idx[:, None], idx[None, :]].ravel()
    a = xp.sort(a[xp.isfinite(a)])
    if a.size == 0:
        return 1.0
    per_row = n_members * n_members + int(a.size)
    batch = max(1, BATCH_ELEMENTS // per_row)
    above = 0
    for start in range(0, sets.shape[0], batch):
        s = sets[start : start + batch]
        b = sim[s[:, :, None], s[:, None, :]].reshape(s.shape[0], -1)
        above += int((greater_p(a, b, xp) > 0.05).sum())
    return above / sets.shape[0]


def sample_values(
    sim: pd.DataFrame,
    programs: dict[str, Sequence[str]],
    replicates: int,
    seeds: Sequence[int],
    device: str = "gpu",
    sets: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Every program's significance value in one sample (`sim`, its averaged
    LR similarity). `seeds[j]` seeds program j's random sets; `sets` (by
    program name) overrides them, e.g. with the R checks' draws."""
    xp = array_module(device)
    values = xp.asarray(sim.to_numpy())
    out = {}
    for (name, members), seed in zip(programs.items(), seeds, strict=True):
        present = np.flatnonzero(sim.index.isin(members))
        k = len(members)
        if sets is not None:
            draws = xp.asarray(sets[name])
        elif k <= values.shape[0]:
            draws = random_sets(values.shape[0], k, replicates, seed, xp)
        else:
            draws = xp.zeros((1, k), dtype=np.int64)
        out[name] = module_value(values, present, k, draws, xp)
    return out
