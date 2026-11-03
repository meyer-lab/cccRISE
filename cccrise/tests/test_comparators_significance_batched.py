"""Batched significance equals `scriabin.module_significance` on the same
random sets (NumPy backend; the GPU backend is checked in rcheck/)."""

import numpy as np
import pandas as pd
import pytest

from cccrise.comparators import scriabin
from cccrise.comparators.significance_batched import greater_p, sample_values


def _similarity(n: int, seed: int, nan_frac: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.round(rng.normal(size=(60, n)), 1)  # rounding creates ties
    sim = np.round(0.5 + 0.5 * np.corrcoef(x, rowvar=False), 3)
    if nan_frac:
        mask = np.triu(rng.random((n, n)) < nan_frac, 1)
        sim[mask | mask.T] = np.nan
    names = [f"l{i}=r{i}" for i in range(n)]
    return pd.DataFrame(sim, index=names, columns=names)


def test_greater_p_matches_port():
    rng = np.random.default_rng(1)
    a = np.sort(np.round(rng.normal(0.6, 0.1, 50), 2))
    b = np.round(rng.normal(0.5, 0.1, (40, 81)), 2)
    np.testing.assert_array_equal(greater_p(a, b, np), scriabin.wilcox_greater_p(a, b))


@pytest.mark.parametrize("nan_frac", [0.0, 0.1])
def test_sample_values_equal_port_on_same_sets(nan_frac):
    sim = _similarity(40, 2, nan_frac)
    rng = np.random.default_rng(3)
    programs = {
        f"IP-{j}": list(rng.choice(sim.index, size, replace=False))
        for j, size in enumerate([2, 5, 9, 17, 40])
    }
    programs["IP-absent"] = ["x=y", "z=w"]
    programs["IP-oversized"] = [*sim.index, "x=y"]  # 41 > 40 LR pairs
    sets = {
        name: np.stack([rng.choice(40, len(m), replace=False) for _ in range(300)])
        if len(m) <= 40
        else np.zeros((1, len(m)), dtype=int)
        for name, m in programs.items()
    }
    ours = sample_values(sim, programs, 300, [0] * len(programs), "cpu", sets)
    for name, members in programs.items():
        expected = (
            scriabin.module_significance(sim, members, sets[name])
            if len(members) <= 40
            else 1.0
        )
        assert ours[name] == expected, name


def test_drawn_sets_are_subsets_without_replacement():
    from cccrise.comparators.significance_batched import random_sets

    s = random_sets(30, 7, 200, 5, np)
    assert s.shape == (200, 7)
    assert all(len(set(row)) == 7 for row in s.tolist())
    assert (s >= 0).all() and (s < 30).all()
