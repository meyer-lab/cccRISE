"""Tests for the granularity ladder (`validation.granularity_ladder`)."""

import numpy as np
import pytest

from ..validation.granularity_ladder import (
    block_means,
    merge_small,
    n_subtypes,
    random_split,
    representability,
    sample_dominance,
    sender_receiver_separation,
    subtype_labels,
    type_components,
)


def test_n_subtypes_respects_floor():
    assert n_subtypes(1000, 8, 100) == 8
    assert n_subtypes(250, 8, 100) == 2
    assert n_subtypes(50, 8, 100) == 1


def test_subtypes_recover_planted_groups_within_each_type():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 40))
    X[:100, :5] += 8  # two expression groups inside type A
    coarse = np.repeat(["A", "B"], 200)
    labels = subtype_labels(type_components(X, coarse), coarse, 2, min_cells=50)
    a = labels[:200]
    assert set(a[:100]) != set(a[100:]) and len(set(a[:100])) == 1
    assert all(lab.startswith("B ") for lab in labels[200:])


def test_random_split_keeps_sizes_within_types():
    labels = np.array(["A 1"] * 3 + ["A 2"] * 5 + ["B"] * 4)
    coarse = np.array(["A"] * 8 + ["B"] * 4)
    shuffled = random_split(labels, coarse, np.random.default_rng(1))
    assert sorted(shuffled) == sorted(labels)
    assert (shuffled[8:] == "B").all()


def test_representability_of_a_block_constant_loading():
    samples = np.repeat(["s1", "s2"], 4)
    types = np.tile(["a", "a", "b", "b"], 2)
    block = np.array([1, 1, 2, 2, 3, 3, 4, 4], float)[:, None]
    noisy = block + np.array([0.5, -0.5] * 4)[:, None]
    t = representability(np.hstack([block, noisy]), samples, types)
    assert t["r2_sample_type"].iloc[0] == pytest.approx(1.0)
    assert t["r2_sample_type"].iloc[1] < 1.0
    assert t["n_blocks"].iloc[0] == 4


def test_sender_receiver_separation():
    s = np.arange(200, dtype=float)[:, None]
    t = sender_receiver_separation(np.hstack([s, s]), np.hstack([s, -s]), ["x"] * 200)
    assert t["spearman"].tolist() == pytest.approx([1.0, -1.0])
    assert t["top_jaccard"].tolist() == [1.0, 0.0]


def test_sample_dominance():
    t = sample_dominance(["a", "a", "a", "b"], ["s1", "s1", "s2", "s2"]).set_index(
        "label"
    )
    assert t.loc["a", "top_sample_share"] == pytest.approx(2 / 3)
    assert t.loc["b", "n_samples"] == 1


def test_merge_small_dissolves_tiny_clusters():
    points = np.r_[np.zeros((10, 1)), np.full((10, 1), 10.0), [[9.0]]]
    clusters = np.r_[np.zeros(10), np.ones(10), [2]].astype(int)
    merged = merge_small(clusters, points, min_cells=5)
    assert merged.tolist() == [0] * 10 + [1] * 11


def test_block_means():
    values = np.array([[1.0], [3.0], [5.0]])
    out = block_means(values, ["s", "s", "s"], ["a", "a", "b"])
    assert out[:, 0].tolist() == [2.0, 2.0, 5.0]
