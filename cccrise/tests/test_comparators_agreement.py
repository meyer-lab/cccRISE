import numpy as np
import pandas as pd
import pytest

from cccrise.comparators.agreement import (
    cell_type_fractions,
    cell_type_overlap,
    celltype_profiles,
    containment_table,
    joint_congruence,
    linear_cka,
    per_column_containment,
    principal_cosines,
    sample_scaled,
    shared_calls,
    subspace_agreement,
    type_eta_squared,
)

RNG = np.random.default_rng(0)


def test_identical_spans_agree_fully():
    a = RNG.normal(size=(40, 3))
    b = a @ RNG.normal(size=(3, 3))  # same span, different basis
    np.testing.assert_allclose(principal_cosines(a, b), 1.0, atol=1e-8)
    np.testing.assert_allclose(per_column_containment(a, b), 1.0, atol=1e-8)
    assert linear_cka(a, a) == pytest.approx(1.0)


def test_orthogonal_column_is_not_contained():
    span = np.eye(6)[:, :2]
    cols = np.eye(6)[:, [0, 4]]
    np.testing.assert_allclose(
        per_column_containment(span, cols), [1.0, 0.0], atol=1e-12
    )


def test_subspace_agreement_flags_shared_span_above_null():
    a = RNG.normal(size=(60, 3))
    b = a[:, :2] + 0.05 * RNG.normal(size=(60, 2))
    out = subspace_agreement(a, b, 200, np.random.default_rng(1)).set_index("statistic")
    assert out.loc["linear_cka", "above_null"]
    assert out.loc["linear_cka", "observed"] > out.loc["linear_cka", "perm_q95"]
    unrelated = subspace_agreement(
        a, RNG.normal(size=(60, 2)), 200, np.random.default_rng(2)
    )
    assert not unrelated.set_index("statistic").loc["linear_cka", "above_null"]


def test_containment_table_and_shared_calls():
    span = RNG.normal(size=(50, 3))
    cols = np.column_stack([span @ [1.0, 0.5, 0.0], RNG.normal(size=50)])
    t = containment_table(span, cols, 200, np.random.default_rng(3))
    assert list(t["contained"]) == [True, False]
    frame = pd.concat(
        [
            t.assign(direction="cccRISE in X", method="PB", rank_mode="own", mode=m)
            for m in ("condition", "lr")
        ]
    )
    calls = shared_calls(frame, ["condition", "lr"])
    assert list(calls["contained_all_modes"]) == [True, False]


def test_joint_congruence_matches_permuted_columns():
    a = {"condition": RNG.normal(size=(10, 3)), "lr": RNG.normal(size=(20, 3))}
    b = {m: v[:, [2, 0, 1]] for m, v in a.items()}
    cong, rows, cols = joint_congruence(a, b, ["condition", "lr"])
    assert list(cols[np.argsort(rows)]) == [1, 2, 0]
    np.testing.assert_allclose(cong[rows, cols], 1.0)


def test_cell_type_fractions():
    types = np.array(["A"] * 10 + ["B"] * 10)
    load = np.column_stack(
        [np.r_[np.ones(10), np.zeros(10)], np.r_[np.zeros(5), np.ones(5), np.zeros(10)]]
    )
    frac = cell_type_fractions(load, types, [0.5])
    a1 = frac[(frac["component"] == 1) & (frac["celltype"] == "A")].iloc[0]
    assert a1["active_frac_0.5"] == 1.0 and a1["loading_share"] == 1.0
    a2 = frac[(frac["component"] == 2) & (frac["celltype"] == "A")].iloc[0]
    assert a2["active_frac_0.5"] == 0.5


def test_type_eta_squared_and_profiles():
    types = np.array(["A", "A", "B", "B"])
    load = np.array([[1.0, 1.0], [1.0, 2.0], [3.0, 1.0], [3.0, 2.0]])
    np.testing.assert_allclose(type_eta_squared(load, types), [1.0, 0.0])
    np.testing.assert_allclose(
        celltype_profiles(-load, types, ["B", "A"]), [[3.0, 1.5], [1.0, 1.5]]
    )


def test_sample_scaling_puts_samples_on_one_scale():
    samples = np.array(["s1"] * 4 + ["s2"] * 16)
    load = np.r_[np.full(4, 0.5), np.full(16, 0.25)][:, None]
    scaled = sample_scaled(load, samples)
    np.testing.assert_allclose(scaled[:, 0], 1.0)  # 0.5*sqrt(4) = 0.25*sqrt(16)


def test_cell_type_overlap_separates_shared_and_distinct_cells():
    types = np.array(["A"] * 10)
    sender = np.r_[np.ones(4), np.zeros(6)][:, None]
    same = cell_type_overlap(sender, sender, types, 0.5).iloc[0]
    assert same["both_frac"] == 0.4 and same["jaccard"] == 1.0
    receiver = np.r_[np.zeros(6), np.ones(4)][:, None]
    distinct = cell_type_overlap(sender, receiver, types, 0.5).iloc[0]
    assert distinct["sender_frac"] == 0.4 and distinct["receiver_frac"] == 0.4
    assert distinct["both_frac"] == 0.0 and distinct["jaccard"] == 0.0
    assert distinct["either_frac"] == 0.8 and same["either_frac"] == 0.4
