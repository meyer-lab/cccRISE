"""Unit tests for the Scriabin seed-agreement port (`comparators.scriabin_programs`,
`runners.run_scriabin_seeds`)."""

import pytest

from cccrise.comparators.scriabin_programs import merge_programs
from cccrise.runners.run_scriabin_seeds import best_match


def test_merge_unites_modules_past_the_threshold_and_keeps_the_rest():
    per_sample = {
        "s1": {"IP-1": ["A", "B", "C"], "IP-2": ["X", "Y"]},
        "s2": {"IP-1": ["A", "B", "D"]},
    }
    merged = merge_programs(per_sample, sim_threshold=0.15)
    assert sorted(map(sorted, merged.values())) == [["A", "B", "C", "D"], ["X", "Y"]]
    assert list(merged) == ["IP-1", "IP-2"]


def test_merge_leaves_modules_at_or_under_the_threshold():
    per_sample = {"s1": {"IP-1": ["A", "B"]}, "s2": {"IP-1": ["C", "D"]}}
    assert len(merge_programs(per_sample)) == 2


def test_best_match_is_symmetric_and_one_for_equal_sets():
    a = [{"A", "B"}, {"C"}]
    b = [{"C"}, {"A", "B", "D"}]
    assert best_match(a, a) == 1.0
    assert best_match(a, b) == pytest.approx(best_match(b, a))
    assert best_match(a, b) == pytest.approx((1 + 2 / 3) / 2)
