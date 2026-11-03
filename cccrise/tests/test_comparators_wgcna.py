"""The cached-nearest-neighbour complete linkage equals the full scan."""

import numpy as np
import pytest

from cccrise.comparators import _wgcna


def _full_scan(dist: np.ndarray):
    """hclust.f's complete linkage by scanning every active pair each step
    (ties: lowest i, then lowest j): the reference the R checks validated."""
    n = dist.shape[0]
    d = np.array(dist, dtype=float)
    np.fill_diagonal(d, np.inf)
    active = np.ones(n, dtype=bool)
    ia, ib, height = np.empty(n - 1, int), np.empty(n - 1, int), np.empty(n - 1)
    for step in range(n - 1):
        masked = np.where(np.triu(np.outer(active, active), 1), d, np.inf)
        i, j = divmod(int(np.argmin(masked)), n)
        ia[step], ib[step], height[step] = i + 1, j + 1, masked[i, j]
        merged = np.maximum(d[i], d[j])
        d[i, :], d[:, i] = merged, merged
        d[i, i] = np.inf
        active[j] = False
        d[j, :], d[:, j] = np.inf, np.inf
    return ia, ib, height


def _symmetric(x: np.ndarray) -> np.ndarray:
    d = np.triu(x, 1)
    return d + d.T


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ties", [False, True])
def test_complete_linkage_equals_full_scan(seed, ties):
    rng = np.random.default_rng(seed)
    n = 60
    x = rng.integers(0, 6, (n, n)).astype(float) if ties else rng.random((n, n))
    dist = _symmetric(x)
    got = _wgcna._complete_linkage_steps(dist)
    want = _full_scan(dist)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)
