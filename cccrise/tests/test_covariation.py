"""Unit tests for cccrise.covariation."""

import numpy as np

from cccrise import covariation


def test_context_scores_weight_senders_and_receivers():
    tensor = np.zeros((2, 2, 2, 3))
    tensor[:, 0, 1, :] = [[1, 2, 3], [4, 5, 6]]
    tensor[:, 1, 0, :] = 100
    scores = covariation.context_scores(tensor, np.array([1, 0]), np.array([0, 2]))
    np.testing.assert_array_equal(scores, [[2, 4, 6], [8, 10, 12]])


def test_top_pairs_are_the_heaviest_by_magnitude():
    assert covariation.top_pairs(np.array([0.1, -3, 2, 0.5]), n=2).tolist() == [1, 2]


def test_carrying_pairs_stop_once_the_share_is_carried():
    lr = np.array([0.1, -3, 2, 1, 0.5])  # squared shares .00, .63, .28, .07, .02
    assert covariation.carrying_pairs(lr, share=0.9).tolist() == [1, 2]
    assert covariation.carrying_pairs(lr, share=0.95).tolist() == [1, 2, 3]
    assert covariation.carrying_pairs(lr, share=0.95, cap=2).tolist() == [1, 2]


def test_baselines_without_associated_pairs_match_those_with():
    rng = np.random.default_rng(0)
    scores = rng.random((12, 40))
    top = np.arange(5)
    associated = np.zeros(40, bool)
    associated[20:30] = True
    full = covariation.covariation_baselines(scores, top, associated)
    plain = covariation.covariation_baselines(scores, top)
    assert list(plain) == ["random", "expression_matched"]
    for name, draws in plain.items():
        np.testing.assert_array_equal(draws, full[name])


def test_gene_sharing_pairs_are_left_out():
    rng = np.random.default_rng(1)
    scores = rng.random((10, 3))
    genes = [{"A", "R1"}, {"B", "R1"}, {"C", "R2"}]
    full = covariation.pairwise_rank_correlations(scores, [0, 1, 2])
    disjoint = covariation.pairwise_rank_correlations(scores, [0, 1, 2], genes)
    assert np.isnan(disjoint[0])  # pairs 0 and 1 share R1
    np.testing.assert_array_equal(disjoint[1:], full[1:])
