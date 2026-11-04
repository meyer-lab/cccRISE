"""The Scriabin seed-stability runner's agreement measures."""

import pytest

from cccrise.runners import run_scriabin_stability as R


def test_best_match_bounds():
    a = [{"x", "y"}, {"z"}]
    assert R.best_match(a, a) == 1.0
    assert R.best_match(a, [{"q"}]) == 0.0
    assert R.best_match([], []) == 1.0
    assert R.best_match(a, []) == 0.0


def test_best_match_is_symmetric_mean_of_both_directions():
    a, b = [{"x", "y"}], [{"x"}, {"y"}]
    # a -> b: best 1/2; b -> a: each 1/2
    assert R.best_match(a, b) == pytest.approx(0.5)
    assert R.best_match(b, a) == R.best_match(a, b)


def test_condition_rows_count_seed_pairs_and_left_out_groups():
    modules = {
        s: {g: {"default": {"IP-1": ["A=B", "C=D"]}} for g in ("g1", "g2", "g3")}
        for s in (1, 2)
    }
    rows = R._condition_rows(modules, ["g1", "g2", "g3"], (1, 2), "default")
    seed = [r for r in rows if r["Measure"] == "seed"]
    loo = [r for r in rows if r["Measure"] == "leave_one_out"]
    assert len(seed) == 1 and seed[0]["Value"] == 1.0
    assert len(loo) == 2 * 3 and all(r["Value"] == 1.0 for r in loo)
