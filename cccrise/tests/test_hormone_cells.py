"""The per-cell quantities of the IBD carrier analysis
(`validation.hormone_cells`): contributions add up to the eigenstate-level
signal, composition shares sum to 1, and the null is calibrated."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cccrise.validation import hormone_cells as H


def _toy(seed=0, n=400, rank=5):
    rng = np.random.default_rng(seed)
    genes = ["L1", "R1", "R2", "L2"]
    X = sparse.csr_matrix(rng.poisson(0.5, (n, len(genes))).astype(float))
    P = rng.standard_normal((n, rank))
    samples = rng.choice(["a", "b", "c"], n)
    return genes, X, P, samples


def test_contribution_adds_up_to_eigenstate_signal():
    genes, X, P, samples = _toy()
    means = pd.Series(np.asarray(X.mean(axis=0)).ravel(), index=genes)
    b = np.random.default_rng(1).standard_normal(P.shape[1])
    weights = pd.Series({"L1": 0.75, "L2": 0.25})
    contribution = (P @ b) * H.centred_signal(X, genes, means, weights)
    dense = X.toarray() - means.to_numpy()
    for s in np.unique(samples):
        rows = samples == s
        eigen = b @ (P[rows].T @ dense[rows])  # per gene, eigenstate level
        expected = 0.75 * eigen[0] + 0.25 * eigen[3]
        assert contribution[rows].sum() == pytest.approx(expected)


def test_complex_takes_min_over_subunits():
    genes, X, _, _ = _toy()
    out = H.token_matrix(X, genes, ["R1&R2", "L1", "MISSING"])
    np.testing.assert_allclose(
        out[:, 0], np.minimum(X[:, 1].toarray(), X[:, 2].toarray()).ravel()
    )
    assert np.isnan(out[:, 2]).all()


def test_side_weights_sum_to_one_and_respect_cut():
    D = np.array([[1.0], [0.5], [0.1]])
    pairs = ["A^X", "B^X", "A^Y"]
    w = H.side_token_weights(D, pairs, 1, "sender")
    assert w.sum() == pytest.approx(1)
    assert w["A"] == pytest.approx((1 + 0.01) / (1 + 0.25 + 0.01))
    cut = H.side_token_weights(D, pairs, 1, "receiver", rho=0.4)
    assert list(cut.index) == ["X"]


def test_composition_shares_sum_to_one():
    _, _, P, samples = _toy()
    labels = np.random.default_rng(2).choice(["x", "y", "z"], len(samples))
    lineage = np.where(labels == "z", "L2", "L1")
    assert H.eigen_mixture(P).sum(axis=1) == pytest.approx(np.ones(len(P)))
    shares = H.sample_equal_shares(P**2, samples, labels, ["x", "y", "z"])
    np.testing.assert_allclose(shares.sum(), 1)
    within = H.sample_equal_shares(
        P**2, samples, labels, ["x", "y", "z"], within=lineage
    )
    np.testing.assert_allclose(within.loc[["x", "y"]].sum(), 1)
    np.testing.assert_allclose(within.loc["z"], 1)


def test_oracle_model_is_exact_on_label_means():
    rng = np.random.default_rng(3)
    samples = rng.choice(["a", "b"], 200)
    labels = rng.choice(["x", "y"], 200)
    v = (
        pd.Series(rng.standard_normal(200))
        .groupby([samples, labels])
        .transform("mean")
        .to_numpy()
    )
    assert H.representability(v, samples, labels) == pytest.approx(1)


def test_direction_null_is_calibrated_for_a_random_direction():
    rng = np.random.default_rng(4)
    n, rank = 2000, 8
    P = rng.standard_normal((n, rank))
    labels = rng.choice(4, n)
    onehot = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(4, n))
    observed = H.random_directions(rank, 1, seed=99)[:, 0]
    _, null, exceed = H.direction_null(
        P, observed, None, onehot, H.random_directions(rank, 200)
    )
    # a random observed direction is exchangeable with the null ones
    assert null.shape == (4, 200)
    assert 0.2 < (exceed / 200).mean() < 0.8


def test_levels():
    assert H.h1_level(0, 0, 0.5, 0.4) == 0
    assert H.h1_level(1, 0, 0.05, 0.01) == 1
    assert H.h1_level(1, 0, 0.12, 0.06) == 2
    assert H.h1_level(0, 1, 0.25, 0.15) == 3
    assert H.h2_level(0.0, 0.0, False) == 0
    assert H.h2_level(0.2, 0.3, True) == 3


def test_bh_matches_definition():
    q = H.bh(np.array([0.01, 0.04, 0.03, 0.5]))
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])
