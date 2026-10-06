import numpy as np
import pandas as pd

from cccrise.validation.eigenstate_celltypes import (
    active,
    bimodality_coefficient,
    classify,
    fine_profile,
    fraction_checks,
    histogram_table,
    home_types,
    orient,
    random_direction_null,
    score_units,
)

RNG = np.random.default_rng(0)
FINE = np.array(["f1"] * 100 + ["f2"] * 100 + ["g1"] * 100)
COARSE = np.array(["A"] * 200 + ["B"] * 100)


def test_orient_makes_the_large_tail_positive():
    x = np.r_[RNG.normal(0, 0.1, 990), -np.full(10, 5.0)][:, None]
    assert orient(x)[-10:, 0].min() > 0


def test_a_program_filling_one_fine_type_is_fine_aligned():
    score = np.r_[np.ones(100), np.zeros(200)]
    s = classify(fine_profile(active(score), FINE, COARSE))
    assert s["call"] == "fine-aligned" and s["top_fine"] == "f1"


def test_a_fraction_of_two_fine_types_is_cross_cutting():
    score = np.r_[
        np.r_[np.ones(50), np.zeros(50)],
        np.r_[np.ones(40), np.zeros(60)],
        np.zeros(100),
    ]
    s = classify(fine_profile(active(score), FINE, COARSE))
    assert s["call"] == "cross-cutting" and s["n_partial_fine"] == 2
    assert s["partial_share"] == 1.0


def test_home_types_and_histogram():
    score = np.r_[np.r_[np.ones(50), np.zeros(150)], np.zeros(100)]
    profile = fine_profile(active(score), FINE, COARSE)
    assert home_types(profile) == ["A"]
    h = histogram_table(score[:200], FINE[:200], 0.5, n_bins=5)
    assert h["count"].sum() == 200 and set(h["fine"]) == {"f1", "f2"}


def test_score_units_and_null_run():
    scores = {"E1": RNG.normal(size=300)}
    summary, profile, hist = score_units(scores, COARSE, FINE, n_bins=5)
    assert list(summary["unit"]) == ["E1"] and set(profile["unit"]) == {"E1"}
    assert set(hist["unit"]) == {"E1"}
    null = random_direction_null(RNG.normal(size=(300, 4)), FINE, COARSE, 10, RNG)
    assert len(null) == 10 and isinstance(null, pd.DataFrame)


def test_fraction_checks_separate_a_subset_from_a_graded_program():
    graded = RNG.normal(1.0, 0.3, 2000)
    two_modes = np.r_[RNG.normal(0, 0.1, 1000), RNG.normal(3, 0.1, 1000)]
    assert bimodality_coefficient(two_modes) > 5 / 9 > bimodality_coefficient(graded)
    fine = np.array(["f"] * 2000)
    subset = fraction_checks(two_modes, fine, "f")
    # A subset's active fraction holds as the cut moves; a gradient's does not.
    assert (
        subset["top_fine_active_frac_tau0.25"] == subset["top_fine_active_frac_tau0.75"]
    )
    gradient = fraction_checks(graded, fine, "f")
    assert (
        gradient["top_fine_active_frac_tau0.25"]
        > gradient["top_fine_active_frac_tau0.75"] + 0.3
    )
