import numpy as np
import pandas as pd

from cccrise.benchmark_null import block_permutations, eta_squared
from cccrise.validation.celltype_conditions import (
    eta_squared_subset,
    per_sample_means,
    unit_maxt,
)

RNG = np.random.default_rng(0)


def test_eta_squared_subset_matches_the_benchmark_eta_squared():
    x = RNG.normal(size=30)
    codes = np.repeat([0, 1, 2], 10)
    perms = block_permutations(np.arange(30), 50, RNG)
    expected = eta_squared(x[:, None], codes, perms)[:, 0]
    np.testing.assert_allclose(eta_squared_subset(x, codes[perms]), expected)


def test_per_sample_means_counts_cells():
    loadings = np.array([[1.0], [3.0], [5.0], [7.0]])
    out = per_sample_means(
        loadings,
        np.array(["s1", "s1", "s2", "s2"]),
        np.array(["a", "a", "a", "b"]),
        np.array(["s1", "s2"]),
    )
    row = out[(out["type"] == "a") & (out["condition"] == "s1")].iloc[0]
    assert row["mean"] == 2.0 and row["n_cells"] == 2


def test_unit_maxt_finds_a_planted_effect_and_skips_small_units():
    n = 40
    codes = np.repeat([0, 1], n // 2)
    perms = block_permutations(np.arange(n), 500, RNG)
    effect = np.r_[np.zeros(n // 2), np.ones(n // 2)] + RNG.normal(0, 0.3, n)
    sparse = np.full(n, np.nan)
    sparse[:5] = 1.0
    units = {
        ("receiver", "x", 1): effect,
        ("receiver", "y", 1): RNG.normal(size=n),
        ("sender", "z", 1): sparse,  # too few samples: not tested
    }
    table = unit_maxt(units, np.arange(n), codes, perms, min_samples=20)
    assert len(table) == 2 and set(table["n_units"]) == {2}
    hit = table[table["key"] == ("receiver", "x", 1)].iloc[0]
    assert hit["maxt_p"] < 0.01 and hit["perm_p"] <= hit["maxt_p"]
    assert isinstance(table, pd.DataFrame)
