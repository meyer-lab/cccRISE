"""Tests for top-k LR-pair agreement (`comparators.lr_concordance`)."""

import numpy as np
import pandas as pd
import pytest

from ..comparators.lr_concordance import (
    abundance_null,
    best_matches,
    classify_components,
    coverage_table,
    head_mass,
    missed_pairs,
    null_thresholds,
    overlap_table,
    ranks,
    shared_counts,
    shared_mass,
    top_k_sets,
    whole_model_percentile,
)


def test_top_k_sets_and_shared_counts():
    lr = np.array([[5, 0], [4, 1], [3, 2], [0, 9], [1, 8]], float)
    sets = top_k_sets(lr, 2)
    assert sets == [frozenset({0, 1}), frozenset({3, 4})]
    assert shared_counts(sets, top_k_sets(lr[:, ::-1], 2)).tolist() == [[0, 2], [2, 0]]


def test_ranks_are_one_based_by_magnitude():
    lr = np.array([[1.0], [-3.0], [2.0]])
    assert ranks(lr)[:, 0].tolist() == [3, 1, 2]


def test_abundance_null_concentrates_on_abundant_pairs():
    rng = np.random.default_rng(0)
    flat = abundance_null(np.ones(100), 10, 2000, rng)
    skewed = abundance_null(np.r_[np.full(10, 1e6), np.ones(90)], 10, 2000, rng)
    assert flat.mean() == pytest.approx(1.0, abs=0.15)  # k^2 / n
    assert skewed.mean() > 9


def test_null_thresholds_columns():
    t = null_thresholds(np.ones(50), (5, 10), 500, np.random.default_rng(1))
    assert t["k"].tolist() == [5, 10]
    assert (t["uniform_q95"] >= t["uniform_mean"]).all()


def test_overlap_and_best_match():
    rng = np.random.default_rng(2)
    ref = rng.random((60, 2))
    other = np.column_stack([rng.random(60), ref[:, 0] + 0.01 * rng.random(60)])
    thresholds = pd.DataFrame({"k": [5], "null_q95": [1.0]})
    table = overlap_table(ref, other, (5,), thresholds)
    best = best_matches(table, k=5).set_index("comparator_component")
    assert best.loc[2, "cccrise_component"] == 1
    assert best.loc[2, "shared"] == 5 and bool(best.loc[2, "hit"])


def test_coverage_detects_one_to_many():
    ref = np.zeros((10, 1))
    ref[:4, 0] = [4, 3, 2, 1]
    other = np.zeros((10, 2))
    other[[0, 1, 8, 9], 0] = [4, 3, 2, 1]
    other[[2, 3, 6, 7], 1] = [4, 3, 2, 1]
    cov = coverage_table(ref, other, (4,))
    row = cov[cov["direction"] == "cccRISE in comparator"].iloc[0]
    assert row["coverage"] == 1.0 and row["best_single"] == 0.5


def test_missed_pairs_and_whole_model_percentile():
    source = np.zeros((60, 1))
    source[59, 0] = 1.0
    target = np.arange(60, 0, -1, dtype=float)[:, None]  # pair 59 ranks last
    table = missed_pairs(source, target, None, 1, [f"p{i}" for i in range(60)])
    assert table["lr_pair"].tolist() == ["p59"]
    assert bool(table["missed"].iloc[0]) and table["best_target_rank"].iloc[0] == 60
    pct = whole_model_percentile(np.array([[1.0, 0], [0, 1.0]]), np.array([3.0, 1.0]))
    assert pct.tolist() == [1.0, 0.5]


def test_classify_components():
    m = pd.DataFrame(
        {
            "hit": [True, True, False, False],
            "sender_cos": [0.9, 0.9, 0.1, 0.1],
            "receiver_cos": [0.9, 0.2, 0.1, 0.1],
            "condition_cos": [0.9, 0.9, 0.1, 0.1],
            "condition_pr": [3.0, 3.0, 1.1, 4.0],
        }
    )
    assert classify_components(m).tolist() == [
        "recovered",
        "same LR, different who/where",
        "not recovered, one sample",
        "not recovered, broad",
    ]


def test_shared_mass_and_head_mass():
    a = np.array([[1.0], [0.9], [0.0], [0.0]])
    b = np.array([[0.1], [0.0], [3.0], [0.2]])  # top-2: pairs 2, 3
    k2 = top_k_sets(a, 2), top_k_sets(b, 2)
    assert shared_mass(*k2, b)[0, 0] == 0.0
    k4 = top_k_sets(a, 4), top_k_sets(b, 4)
    assert shared_mass(*k4, b)[0, 0] == pytest.approx(1.0)
    hm = head_mass(b, (1, 2)).set_index("k")["head_mass"]
    assert hm[1] == pytest.approx(9 / 9.05)
