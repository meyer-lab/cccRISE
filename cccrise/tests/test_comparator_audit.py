"""The comparator audit's per-column profiles and pre-registered classes."""

import numpy as np
import pandas as pd

from cccrise.benchmark import ComparatorFit
from cccrise.runners.run_comparator_audit import classify, column_profiles


def _fit(core=None, **modes):
    return ComparatorFit(method="m", rank_mode="own", rank="r", condition_factor=modes.pop("condition"),
                         details={}, modes=modes, weights=None, core=core)  # fmt: skip


def test_column_profiles_of_a_tucker_column_are_its_core_slice_mapped():
    rng = np.random.default_rng(0)
    a, b, c = (np.linalg.qr(rng.standard_normal((n, 3)))[0] for n in (12, 6, 5))
    u, v, w = (rng.standard_normal(3) for _ in range(3))
    core = np.zeros((3, 2, 3, 3))
    core[:, 1] = np.einsum("i,j,k->ijk", u, v, w)  # LR column 2 is rank one
    fit = _fit(core=core, condition=a, sender=b, receiver=c, lr=np.eye(4, 2))
    profiles = column_profiles(fit, 1)
    for name, factor, vector in (("sample", a, u), ("sender", b, v), ("receiver", c, w)):
        expected = factor @ vector
        assert abs(abs(np.corrcoef(profiles[name], expected)[0, 1]) - 1) < 1e-10, name


def test_classify_takes_the_first_matching_class():
    types = ["T", "B"]
    fold = pd.DataFrame({"L": [10.0, 0.1], "R": [10.0, 0.1]}, index=types)
    detected = pd.DataFrame({"L": [0.9, 0.0], "R": [0.9, 0.0]}, index=types)
    sample = np.arange(8.0)
    fractions = pd.DataFrame({"T": np.zeros(8) + 0.5, "B": np.r_[1, 0, 1, 0, 1, 0, 1, 0.0]})
    row = {"effective_samples": 1.2, "top_pair": "L^R", "top_sender": "T", "top_receiver": "B"}
    assert classify(row, fold, detected, fractions, sample) == "one sample"
    row["effective_samples"] = 5.0
    assert classify(row, fold, detected, fractions, sample) == "marker or homophilic pair"
    row["top_sender"] = "B"
    fractions["B"] = sample
    assert classify(row, fold, detected, fractions, sample) == "abundance readout"
    fractions["B"] = np.r_[1, 0, 1, 0, 1, 0, 1, 0.0]
    assert classify(row, fold, detected, fractions, sample) == "candidate biology"
