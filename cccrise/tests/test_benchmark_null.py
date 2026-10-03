"""Unit tests for cccrise.benchmark_null."""

import anndata
import numpy as np
import pandas as pd
import pytest

from cccrise.anova import anova_component_associations
from cccrise.benchmark_null import (
    association_labels,
    association_permutations,
    block_permutations,
    eta_squared,
    maxt_null,
    paired_difference,
    summarize_null,
)


def test_block_permutations_keep_patients_together():
    patients = np.array(["a", "b", "b", "c", "d", "d", "e"])
    perms = block_permutations(patients, 200, np.random.default_rng(0))
    assert perms.shape == (200, 7)
    for perm in perms:
        assert sorted(perm) == list(range(7))
        # A two-condition patient's positions are filled from one patient's pair
        for pair in ([1, 2], [4, 5]):
            assert set(perm[pair]) in ({1, 2}, {4, 5})
    # Both pairs, and both within-pair orders, actually occur
    assert len({tuple(perm[[1, 2]]) for perm in perms}) == 4


def test_eta_squared_matches_the_anova():
    rng = np.random.default_rng(1)
    loadings = rng.normal(size=(20, 3))
    labels = pd.Series(["x"] * 7 + ["y"] * 6 + ["z"] * 7)
    X = anndata.AnnData(shape=(20, 0))
    X.uns["A"] = loadings
    expected = anova_component_associations(X, labels)["EtaSquared"].to_numpy()
    _, codes = np.unique(labels, return_inverse=True)
    observed = eta_squared(loadings, codes, np.arange(20)[None, :])[0]
    np.testing.assert_allclose(observed, expected)


def test_maxt_null_is_calibrated_and_detects_signal():
    rng = np.random.default_rng(2)
    n = 30
    codes = np.repeat([0, 1], n // 2)
    perms = block_permutations(np.arange(n), 999, rng)
    noise = rng.normal(size=(n, 4))
    _, p_noise, _ = maxt_null(noise, np.arange(n), codes, perms)
    assert np.all(p_noise > 0.01)

    signal = noise.copy()
    signal[:, 2] += 3.0 * codes
    observed, p_adj, null_max = maxt_null(signal, np.arange(n), codes, perms)
    assert p_adj[2] == pytest.approx(1 / 1000)
    assert null_max.shape == (999,)
    summary = summarize_null(observed, p_adj, null_max)
    assert summary["BestComponent"] == 3
    assert summary["NullN"] == 999


def test_maxt_null_is_nan_when_a_group_is_too_small():
    codes = np.array([0, 0, 0, 1])
    perms = block_permutations(np.arange(4), 10, np.random.default_rng(0))
    observed, p_adj, null_max = maxt_null(np.ones((4, 2)), np.arange(4), codes, perms)
    assert np.all(np.isnan(observed)) and np.all(np.isnan(p_adj))
    assert np.isnan(summarize_null(observed, p_adj, null_max)["PermP"])


def test_association_permutations_drop_unkept_conditions():
    obs = pd.DataFrame(
        {
            "sample": ["s1", "s2", "s3", "s4", "s5"],
            "status": ["control", "declined", "recovered", "control", "declined"],
        }
    )
    X = anndata.AnnData(obs=obs)
    labels = association_labels(
        X,
        "sample",
        {"group_column": "status", "levels": ["control", "declined"]},
    )
    kept, codes, perms = association_permutations(
        labels,
        pd.Series(obs["sample"].to_numpy()),
        5,
        np.random.default_rng(0),
    )
    assert list(kept) == [0, 1, 3, 4]
    assert list(codes) == [0, 1, 0, 1]
    assert perms.shape == (5, 4)


def test_paired_difference_is_centred_on_the_null_not_zero():
    rng = np.random.default_rng(3)
    null_a = rng.normal(0.10, 0.02, size=999)
    null_b = null_a - 0.05  # A always leads by 0.05 under the null
    at_null = paired_difference(0.30, null_a, 0.25, null_b)
    assert at_null["NullDiffMean"] == pytest.approx(0.05)
    assert at_null["PairedP"] == pytest.approx(1.0)

    ahead = paired_difference(0.30, null_a, 0.10, null_b + rng.normal(0, 0.01, 999))
    assert ahead["Difference"] == pytest.approx(0.20)
    assert ahead["PairedP"] == pytest.approx(1 / 1000)
    assert ahead["PairedPTwoSided"] == pytest.approx(1 / 1000)
    assert ahead["NullDiffQ025"] < ahead["NullDiffMean"] < ahead["NullDiffQ975"]


def test_paired_difference_is_nan_without_a_null():
    out = paired_difference(np.nan, np.full(5, np.nan), 0.1, np.zeros(5))
    assert np.isnan(out["PairedP"])
