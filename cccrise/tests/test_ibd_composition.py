"""The IBD composition runner's statistics."""

import numpy as np
from scipy import stats

from cccrise.runners.run_ibd_composition import (
    _dummies,
    _group_stat,
    _residualise,
    clr_composition,
)


def test_clr_composition_is_centred_both_ways():
    counts = np.random.default_rng(0).integers(0, 50, size=(12, 5)).astype(float)
    clr = clr_composition(counts)
    np.testing.assert_allclose(clr.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(clr.sum(axis=1), clr.sum(axis=1).mean(), atol=1e-12)


def test_group_stat_matches_a_two_sample_t_and_one_way_f():
    rng = np.random.default_rng(1)
    labels = np.repeat(np.array(["a", "b", "c"]), 10)
    y = rng.standard_normal((30, 4)) + (labels == "a")[:, None]
    ones = np.ones((30, 1))
    two = labels != "c"
    residual = _residualise(y[two], ones[two])
    t = _group_stat(residual, _residualise(_dummies(labels[two], ("a", "b")), ones[two]), 20 - 2)
    expected = np.abs(stats.ttest_ind(y[two][labels[two] == "a"], y[two][labels[two] == "b"]).statistic)
    np.testing.assert_allclose(t, expected)
    f = _group_stat(_residualise(y, ones), _residualise(_dummies(labels, ("a", "b", "c")), ones), 30 - 3)
    np.testing.assert_allclose(f, stats.f_oneway(*(y[labels == g] for g in "abc")).statistic)
