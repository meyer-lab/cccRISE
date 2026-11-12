"""Scriabin's interaction-program merge (BlishLab/scriabin, R, pinned at
``bce695b``; ``FindAllInteractionPrograms``), as PR #275's port has it
(`cccrise.comparators.scriabin` there): every sample's LR modules pooled and,
while the largest Jaccard similarity below 1 exceeds the threshold, every
module of a pair achieving it replaced by their union. The Jaccard matrix is
updated one row per merge step rather than recomputed, so IBD's thousands of
modules merge quickly; values, ties and order are upstream's, so the output
is too. Used by `runners.run_scriabin_seeds`."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

#: ``InteractionPrograms``' default: groups of more cells are subsampled.
ITERATE_THRESHOLD = 300
#: ``FindAllInteractionPrograms``' default merge Jaccard.
SIM_THRESHOLD = 0.15


class _IncrementalJaccard:
    """Module x module Jaccard similarity (ade4's ``1 - dist.binary(method =
    1)^2``, entry by entry) with each row's largest value below 1 cached, so
    a merge step updates one row instead of recomputing the matrix. Slots are
    reused: a merged module takes the slot of one it replaces, and `order`
    holds each slot's position in upstream's list (merged modules go last),
    which fixes the tie order."""

    def __init__(self, members: list[list[str]]):
        universe = pd.unique(np.concatenate([np.asarray(m) for m in members]))
        self.index = {lr: i for i, lr in enumerate(universe)}
        n = len(members)
        self.member = np.zeros((n, len(universe)))
        for i, m in enumerate(members):
            self.member[i, [self.index[x] for x in m]] = 1.0
        self.sizes = self.member.sum(axis=1)
        self.alive = np.ones(n, dtype=bool)
        self.order = np.arange(n)
        shared = self.member @ self.member.T
        self.sim = self._formula(
            shared, self.sizes[:, None] + self.sizes[None, :] - shared
        )
        self.rowmax = np.full(n, -np.inf)
        self.argmax = np.zeros(n, dtype=np.int64)
        self._refresh(np.arange(n))

    @staticmethod
    def _formula(shared: np.ndarray, union: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(1.0 - shared / union) ** 2

    def _refresh(self, rows: np.ndarray, chunk: int = 2048) -> None:
        """Recompute the cached maxima of `rows` over alive columns."""
        for start in range(0, len(rows), chunk):
            r = rows[start : start + chunk]
            block = np.where(
                self.alive[None, :] & (self.sim[r] < 1), self.sim[r], -np.inf
            )
            self.argmax[r] = block.argmax(axis=1)
            self.rowmax[r] = block[np.arange(len(r)), self.argmax[r]]

    def best(self) -> float:
        return float(self.rowmax[self.alive].max()) if self.alive.sum() > 1 else -np.inf

    def hits(self, value: float) -> np.ndarray:
        """Row slots of every (row, col) pair at `value`, in the column-major
        order of ``which(d == value, arr.ind = TRUE)`` (repeats kept)."""
        rows = np.flatnonzero(self.alive & (self.rowmax == value))
        pairs = [
            (i, j)
            for i in rows
            for j in np.flatnonzero(self.alive & (self.sim[i] == value))
        ]
        pairs.sort(key=lambda p: (self.order[p[1]], self.order[p[0]]))
        return np.array([i for i, _ in pairs])

    def replace(self, removed: np.ndarray, members: list[str], order: int) -> int:
        """Remove the `removed` slots and add `members` (at list position
        `order`) in the first of them; returns its slot."""
        removed = np.unique(removed)
        slot = int(removed[0])
        self.alive[removed] = False
        self.alive[slot] = True
        self.order[slot] = order
        self.member[slot] = 0.0
        self.member[slot, [self.index[x] for x in members]] = 1.0
        self.sizes[slot] = len(members)
        shared = self.member @ self.member[slot]
        col = self._formula(shared, self.sizes + self.sizes[slot] - shared)
        self.sim[slot, :] = col
        self.sim[:, slot] = col
        stale = self.alive & np.isin(self.argmax, removed)
        stale[slot] = True
        self._refresh(np.flatnonzero(stale))
        fresh = self.alive & ~stale & (col < 1) & (col > self.rowmax)
        self.rowmax[fresh], self.argmax[fresh] = col[fresh], slot
        return slot


def _numbered(members: list[list[str]]) -> dict[str, list[str]]:
    return {f"IP-{k + 1}": [str(x) for x in v] for k, v in enumerate(members)}


def merge_programs(
    per_sample: Mapping[str, Mapping[str, list[str]]],
    sim_threshold: float = SIM_THRESHOLD,
) -> dict[str, list[str]]:
    """``FindAllInteractionPrograms``' merge of every sample's modules,
    renamed ``IP-1..`` in upstream's order."""
    members = [list(v) for m in per_sample.values() for v in m.values()]
    if len(members) < 2:
        return _numbered(members)
    jac = _IncrementalJaccard(members)
    next_order = len(members)
    while np.isfinite(value := jac.best()) and value > sim_threshold:
        rows = jac.hits(value)
        merged = list(pd.unique(np.concatenate([members[i] for i in rows])))
        members[jac.replace(rows, merged, next_order)] = merged
        next_order += 1
    alive = np.flatnonzero(jac.alive)
    return _numbered([members[i] for i in alive[np.argsort(jac.order[alive])]])
