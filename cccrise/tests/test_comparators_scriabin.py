"""Unit tests for cccrise.comparators.scriabin and its WGCNA ports.

The R-vs-Python equivalence checks live in rcheck/ (see
test_comparators_rcheck.py); these tests pin down the formulas and the
upstream conventions independently of R.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

from cccrise.comparators import _wgcna, scriabin
from cccrise.comparators.inputs import entity_expression, lr_expression


def test_ccim_is_sqrt_product_with_sender_fastest_rows():
    lig = np.array([[1.0, 4.0], [9.0, 0.0]])  # 2 senders x 2 LR
    rec = np.array([[4.0, 1.0], [1.0, 1.0], [0.0, 9.0]])  # 3 receivers
    m = scriabin.ccim(lig, rec)
    assert m.shape == (6, 2)
    for r in range(3):
        for s in range(2):
            np.testing.assert_allclose(m[s + 2 * r], np.sqrt(lig[s] * rec[r]))


def test_lr_similarity_drops_all_zero_columns():
    m = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0], [3.0, 0.0, 0.0]])
    keep, sim = scriabin.lr_similarity(m)
    assert keep.tolist() == [True, False, True]
    np.testing.assert_allclose(sim, 0.5 + 0.5 * np.corrcoef(m[:, [0, 2]], rowvar=False))


def test_ccim_similarity_equals_direct():
    rng = np.random.default_rng(3)
    lig = rng.gamma(0.5, size=(40, 12)) * (rng.random((40, 12)) < 0.4)
    rec = rng.gamma(0.5, size=(40, 12)) * (rng.random((40, 12)) < 0.4)
    lig[:, 3] = 0.0  # an all-zero column is dropped
    keep, sim = scriabin.ccim_similarity(lig, rec)
    keep_d, sim_d = scriabin.lr_similarity(scriabin.ccim(lig, rec))
    assert keep.tolist() == keep_d.tolist()
    np.testing.assert_allclose(sim, sim_d, rtol=0, atol=1e-12)


def test_averaged_similarity_streamed_equals_stacked(monkeypatch):
    rng = np.random.default_rng(5)
    lig = rng.gamma(0.5, size=(80, 10)) * (rng.random((80, 10)) < 0.3)
    rec = rng.gamma(0.5, size=(80, 10)) * (rng.random((80, 10)) < 0.3)
    names = [f"L{i}=R{i}" for i in range(10)]
    subs = [rng.choice(80, 12, replace=False) for _ in range(9)]
    streamed = scriabin.averaged_similarity(lig, rec, names, subs, factored=True)
    monkeypatch.setattr(scriabin, "STREAM_ABOVE", 100)
    stacked = scriabin.averaged_similarity(lig, rec, names, subs, factored=True)
    assert streamed.index.equals(stacked.index)
    np.testing.assert_allclose(streamed, stacked, rtol=0, atol=1e-14)


def test_stratified_subsample_quota_and_cap():
    types = np.array(["a"] * 900 + ["b"] * 90 + ["c"] * 2)
    picks = scriabin.stratified_subsample(types, np.random.default_rng(0))
    counts = pd.Series(types[picks]).value_counts()
    # round(300 * n / N) per type, at least 3, at most the type's size
    assert counts["a"] == round(300 * 900 / 992)
    assert counts["b"] == round(300 * 90 / 992)
    assert counts["c"] == 2
    assert len(np.unique(picks)) == len(picks)


@pytest.mark.parametrize(
    "n, expected", [(250, 0), (300, 0), (301, 1), (885, 2), (3000, 3)]
)
def test_n_iterations(n, expected):
    assert scriabin.n_iterations(n) == expected


def test_wilcox_matches_scipy_asymptotic_with_ties():
    rng = np.random.default_rng(1)
    a = np.round(rng.normal(0.6, 0.1, 40), 2)
    b = np.round(rng.normal(0.5, 0.1, (5, 60)), 2)  # rounding creates ties
    ours = scriabin.wilcox_greater_p(a, b)
    ref = [
        mannwhitneyu(a, row, alternative="greater", method="asymptotic").pvalue
        for row in b
    ]
    np.testing.assert_allclose(ours, ref, rtol=1e-10)


def test_merge_programs_unions_overlapping_modules():
    per_sample = {
        "s1": {"IP-1": ["a=x", "b=x", "c=y"], "IP-2": ["d=z", "e=z"]},
        "s2": {"IP-1": ["a=x", "b=x", "f=y"], "IP-2": ["g=w", "h=w"]},
    }
    merged = scriabin.merge_programs(per_sample)
    sets = sorted(sorted(v) for v in merged.values())
    assert sets == [["a=x", "b=x", "c=y", "f=y"], ["d=z", "e=z"], ["g=w", "h=w"]]


def test_merge_programs_keeps_identical_modules_apart():
    # upstream only merges pairs with Jaccard strictly below 1
    per_sample = {"s1": {"IP-1": ["a=x", "b=x"]}, "s2": {"IP-1": ["a=x", "b=x"]}}
    assert len(scriabin.merge_programs(per_sample)) == 2


def test_module_significance_is_one_for_absent_module():
    sim = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
    sets = np.array([[0, 1]])
    assert scriabin.module_significance(sim, ["x=y", "z=w"], sets) == 1.0


def test_kept_programs_requires_two_ligands_and_receptors():
    modules = {"IP-1": ["a=x", "b=y"], "IP-2": ["a=x", "a=y"]}
    sig = pd.DataFrame({"s1": [0.0, 0.0]}, index=list(modules))
    assert list(scriabin.kept_programs(modules, sig)) == ["IP-1"]


def test_module_score_is_feature_mean_minus_control_mean():
    X = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 0.0, 6.0]])
    genes = pd.Index(list("abcd"))
    score = scriabin.module_score(X, genes, ["a", "b"], ["c", "d"])
    np.testing.assert_allclose(score, [1.5 - 3.5, 1.0 - 3.0])


def test_complex_expression_uses_min_and_product():
    X = np.array([[2.0, 3.0, 5.0], [0.0, 4.0, 1.0]])
    genes = ["L", "Ra", "Rb"]
    lr = pd.DataFrame({"ligand": ["L"], "receptor": ["Ra&Rb"]})
    lig, rec = lr_expression(X, genes, lr)
    np.testing.assert_allclose(lig[:, 0], [2.0, 0.0])
    np.testing.assert_allclose(rec[:, 0], [3.0, 1.0])
    prod = entity_expression(X, genes, ["Ra&Rb"], agg="product")
    np.testing.assert_allclose(prod[:, 0], [15.0, 4.0])
    missing = entity_expression(X, genes, ["Ra&Rc"])
    np.testing.assert_allclose(missing[:, 0], 0.0)


def _random_dist(n: int, seed: int) -> np.ndarray:
    pts = np.random.default_rng(seed).normal(size=(n, 3))
    return np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))


def test_hclust_heights_match_scipy_complete_linkage():
    dist = _random_dist(40, 2)
    tree = _wgcna.hclust_complete(dist)
    ref = linkage(squareform(dist, checks=False), method="complete")
    np.testing.assert_allclose(np.sort(tree.height), np.sort(ref[:, 2]))
    assert sorted(tree.order.tolist()) == list(range(1, 41))


def test_cutree_height_groups_by_first_appearance():
    dist = np.array([[0, 1, 9, 9], [1, 0, 9, 9], [9, 9, 0, 2], [9, 9, 2, 0]], float)
    tree = _wgcna.hclust_complete(dist)
    assert _wgcna.cutree_height(tree, 5).tolist() == [1, 1, 2, 2]
    assert _wgcna.cutree_height(tree, 0.5).tolist() == [1, 2, 3, 4]


def test_signed_tom_has_unit_diagonal_and_is_symmetric():
    rng = np.random.default_rng(3)
    sim = 0.5 + 0.5 * np.corrcoef(rng.normal(size=(50, 12)), rowvar=False)
    tom = _wgcna.signed_tom(sim**2)
    np.testing.assert_allclose(np.diag(tom), 1.0)
    np.testing.assert_allclose(tom, tom.T)
    assert (tom >= 0).all() and (tom <= 1).all()


def test_cut_number_gives_equal_count_bins():
    x = np.arange(48, dtype=float)
    bins = _wgcna.cut_number(x, 4)
    assert np.bincount(bins)[1:].tolist() == [12, 12, 12, 12]


def test_n_iterate_overrides_but_small_samples_use_full_ccim():
    assert scriabin.n_iterations(3000, n_iterate=128) == 128
    assert scriabin.n_iterations(250, n_iterate=128) == 0


def test_draw_subsamples_follows_settings():
    types = np.array(["a"] * 600 + ["b"] * 300)
    rng = np.random.default_rng(0)
    assert len(scriabin.draw_subsamples(types, rng)) == 2  # round(sqrt(3))
    many = scriabin.ProgramSettings(n_iterate=5)
    assert len(scriabin.draw_subsamples(types, rng, many)) == 5
    full = scriabin.ProgramSettings(iterate_threshold=1000)
    (only,) = scriabin.draw_subsamples(types, rng, full)
    assert only.tolist() == list(range(900))


def _flat_similarity(n_lr: int = 30, seed: int = 4) -> np.ndarray:
    """A near-flat signed similarity, like the ALAD networks (cor near 0)."""
    x = np.random.default_rng(seed).normal(size=(400, n_lr))
    return 0.5 + 0.5 * np.corrcoef(x, rowvar=False)


def test_soft_power_fallback_and_cap():
    sim = _flat_similarity()
    power, _ = _wgcna.pick_soft_threshold(sim, r2_cut=scriabin.R2_CUTOFF)
    fixed = scriabin.ProgramSettings(soft_power=6)
    assert scriabin.choose_soft_power(sim, fixed) == 6
    uncapped = scriabin.ProgramSettings(max_soft_power=None)
    assert scriabin.choose_soft_power(sim, uncapped) == (1 if power is None else power)
    if power is not None and power > scriabin.MAX_SOFT_POWER:
        assert scriabin.choose_soft_power(sim) == 1


def test_upstream_settings_are_the_defaults():
    sim = pd.DataFrame(_flat_similarity())
    sim.index = sim.columns = [f"l{i}=r{i}" for i in range(len(sim))]
    default = scriabin.modules_from_similarity(sim)
    explicit = scriabin.modules_from_similarity(sim, scriabin.ProgramSettings())
    assert default.modules == explicit.modules
    assert default.soft_power == explicit.soft_power


def test_cut_height_quantile_changes_the_cut():
    sim = pd.DataFrame(_flat_similarity(60))
    sim.index = sim.columns = [f"l{i}=r{i}" for i in range(len(sim))]
    labels = {
        q: scriabin.modules_from_similarity(
            sim, scriabin.ProgramSettings(cut_height_quantile=q, min_module_size=3)
        ).labels
        for q in (0.1, 0.9)
    }
    assert (labels[0.1] == 0).sum() >= (labels[0.9] == 0).sum()


@pytest.mark.parametrize("threshold", [0.05, 0.15, 0.3])
def test_incremental_merge_equals_reference(threshold):
    """Same output, ties and order as `merge_programs` (which the R checks
    compare with upstream), on random module sets with many ties."""
    rng = np.random.default_rng(int(threshold * 100))
    for _ in range(60):
        universe = [f"l{i}=r{i}" for i in range(rng.integers(8, 40))]
        per_sample = {
            f"s{s}": {
                f"IP-{k + 1}": list(
                    rng.choice(universe, rng.integers(2, 9), replace=False)
                )
                for k in range(rng.integers(1, 6))
            }
            for s in range(rng.integers(1, 8))
        }
        assert scriabin.merge_programs_incremental(
            per_sample, threshold
        ) == scriabin.merge_programs(per_sample, threshold)
