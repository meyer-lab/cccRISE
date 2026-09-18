"""Unit tests for cccrise.benchmarks.auc."""

import numpy as np
import pandas as pd
import pytest

from cccrise.benchmarks.auc import (
    CLASSIFIERS,
    AUCResult,
    cv_auc,
    encode_labels,
)


def _separable(n: int = 40, n_features: int = 3, seed: int = 0):
    """A score matrix whose first column separates the two classes cleanly."""
    rng = np.random.default_rng(seed)
    y = np.array(["control"] * (n // 2) + ["alad"] * (n // 2))
    signal = np.concatenate([np.full(n // 2, -3.0), np.full(n // 2, 3.0)])
    X = rng.normal(size=(n, n_features)) * 0.1
    X[:, 0] += signal
    return X, y


def _noise(n: int = 40, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.array(["control"] * (n // 2) + ["alad"] * (n // 2))
    return rng.normal(size=(n, n_features)), y


# --------------------------------------------------------------------------
# encode_labels
# --------------------------------------------------------------------------


def test_encode_labels_is_alphabetical_matching_binarize_group():
    y, classes = encode_labels(["control", "alad", "control"])
    # Alphabetical, so "alad" is class 0 -- the same convention pandas'
    # categorical codes give `plotGeneral.binarize_group`.
    assert classes == ("alad", "control")
    assert list(y) == [1, 0, 1]


def test_encode_labels_rejects_missing_and_non_binary():
    with pytest.raises(ValueError, match="no group label"):
        encode_labels(pd.Series(["control", None, "alad"]))
    with pytest.raises(ValueError, match="exactly two label levels"):
        encode_labels(["control", "declined", "recovered"])


# --------------------------------------------------------------------------
# cv_auc -- the statistic
# --------------------------------------------------------------------------


def test_perfectly_separable_scores_give_auc_one():
    X, y = _separable()
    res = cv_auc(X, y, n_repeats=2, n_boot=200, random_state=0)
    assert isinstance(res, AUCResult)
    assert res.auc == pytest.approx(1.0)
    assert res.accuracy == pytest.approx(1.0)
    assert res.auc_lo <= res.auc <= res.auc_hi


def test_pure_noise_lands_near_chance():
    X, y = _noise(n=60, seed=3)
    res = cv_auc(X, y, n_repeats=3, n_boot=200, random_state=1)
    assert 0.3 < res.auc < 0.7


def test_permutation_null_is_near_chance_and_p_is_significant_when_separable():
    X, y = _separable(n=30)
    res = cv_auc(X, y, n_repeats=2, n_permutations=20, n_boot=100, random_state=0)
    assert res.null_auc is not None and res.null_aucs is not None
    assert 0.2 < res.null_auc < 0.8
    # 20 permutations floors the p-value at 1/21.
    assert res.null_p == pytest.approx(1 / 21)
    assert res.null_sd is not None and res.null_sd > 0


def test_null_draws_are_the_same_statistic_as_the_observed_auc_by_default():
    # The p-value compares the observed AUC against the permutation draws, so a
    # draw has to be formed the same way: a mean over the same number of outer
    # CV repeats. A single-repeat draw has larger variance than a mean of five,
    # which fattens the null's upper tail and silently inflates p.
    X, y = _noise(n=30, seed=1)
    res = cv_auc(X, y, n_repeats=3, n_permutations=4, n_boot=0, random_state=0)
    assert res.params["null_n_repeats"] == 3
    assert res.params["null_matched"] is True
    assert res.to_row()["NullNRepeats"] == 3


def test_unmatched_null_is_allowed_but_flagged():
    # Still available as a deliberate cost trade for wide sweeps; it must be
    # visible in the params and on the row so a stale CSV can be recognized.
    X, y = _noise(n=30, seed=1)
    res = cv_auc(
        X, y, n_repeats=3, n_permutations=4, null_n_repeats=1, n_boot=0, random_state=0
    )
    assert res.params["null_n_repeats"] == 1
    assert res.params["null_matched"] is False
    assert res.to_row()["NullNRepeats"] == 1


def test_result_is_invariant_to_row_order():
    X, y = _separable(n=40, seed=5)
    a = cv_auc(X, y, n_repeats=2, n_boot=0, random_state=0)
    perm = np.random.default_rng(11).permutation(len(y))
    b = cv_auc(X[perm], y[perm], n_repeats=2, n_boot=0, random_state=0)
    # Same data, different row order: the pooled out-of-fold AUC must not care.
    assert a.auc == pytest.approx(b.auc, abs=1e-12)


def test_is_deterministic_for_a_fixed_seed():
    X, y = _noise(n=40, seed=7)
    a = cv_auc(X, y, n_repeats=2, n_boot=100, random_state=3)
    b = cv_auc(X, y, n_repeats=2, n_boot=100, random_state=3)
    assert a.auc == b.auc
    assert a.auc_lo == b.auc_lo and a.auc_hi == b.auc_hi


# --------------------------------------------------------------------------
# cv_auc -- the cross-validation contract
# --------------------------------------------------------------------------


def test_every_condition_is_predicted_exactly_once_per_repeat():
    X, y = _noise(n=30, seed=2)
    res = cv_auc(X, y, n_repeats=3, n_boot=0, random_state=0)
    # An out-of-fold matrix with any NaN means some condition was never held
    # out, i.e. the pooled AUC would be scoring a prediction on training data.
    assert res.oof_scores.shape == (3, 30)
    assert np.isfinite(res.oof_scores).all()


def test_grouped_folds_never_split_a_group():
    # Two rows per "patient": ungrouped folds would put copies of one patient on
    # both sides of a split. ALAD does not need this; IBDverse does.
    X, y = _separable(n=40, seed=4)
    # _separable already labels the first half "control"; pairing consecutive
    # rows means every patient's two rows carry the same label.
    groups = np.repeat(np.arange(20), 2)
    res = cv_auc(X, y, groups=groups, n_repeats=2, n_boot=0, random_state=0)
    assert res.params["grouped"] is True
    assert np.isfinite(res.oof_scores).all()


def test_fold_count_is_capped_by_the_rarer_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = np.array(["alad"] * 27 + ["control"] * 3)
    res = cv_auc(X, y, n_splits=10, n_repeats=1, n_boot=0, random_state=0)
    # Only 3 controls, so 10 stratified folds are impossible.
    assert res.params["n_splits"] == 3
    assert res.params["n_splits_requested"] == 10


# --------------------------------------------------------------------------
# cv_auc -- input validation and the classifier plug point
# --------------------------------------------------------------------------


def test_rejects_misaligned_labels_and_nonfinite_scores():
    X, y = _noise(n=20)
    # Drop rows from the score matrix, not the labels, so both classes survive
    # and the alignment check is what fires rather than the two-level check.
    with pytest.raises(ValueError, match="aligned condition-for-condition"):
        cv_auc(X[:15], y)
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        cv_auc(bad, y)


def test_rejects_unknown_classifier():
    X, y = _noise(n=20)
    with pytest.raises(ValueError, match="Unknown classifier"):
        cv_auc(X, y, classifier="random_forest")


@pytest.mark.parametrize("name", sorted(CLASSIFIERS))
def test_every_registered_classifier_runs_and_separates(name):
    # The D1 decision is meant to be swappable: each entry must be usable
    # through the same call, with no method-specific handling.
    X, y = _separable(n=40, n_features=12, seed=1)
    res = cv_auc(X, y, classifier=name, n_repeats=1, n_boot=0, random_state=0)
    assert res.auc > 0.9
    assert res.params["classifier"] == name


def test_accepts_a_dataframe_and_a_series():
    X, y = _separable(n=30)
    res = cv_auc(
        pd.DataFrame(X, columns=[f"Cmp{i + 1}" for i in range(X.shape[1])]),
        pd.Series(y),
        n_repeats=1,
        n_boot=0,
    )
    assert res.n_components == X.shape[1]
    assert res.n_conditions == 30


def test_single_component_matrix_is_scored():
    # CCC-RISE runs at CP rank 3 on ALAD; a rank-1 arm of the condition sweep
    # must still produce a number rather than failing on shape.
    X, y = _separable(n=30, n_features=1)
    res = cv_auc(X, y, n_repeats=1, n_boot=0)
    assert res.n_components == 1
    assert res.auc == pytest.approx(1.0)


def test_inner_cv_never_asks_for_more_folds_than_the_training_class_allows(recwarn):
    # A 5/5/5-style arm of the condition sweep: few minority members, so an
    # uncapped inner CV would build single-class inner folds and select `C`
    # off a scoring grid full of NaNs.
    X, y = _separable(n=16, seed=9)
    cv_auc(X, y, n_splits=4, inner_splits=5, n_repeats=1, n_boot=0, random_state=0)
    messages = [str(w.message) for w in recwarn]
    assert not [m for m in messages if "less than n_splits" in m], messages
    assert not [m for m in messages if "Only one class is present" in m], messages


def test_to_row_and_mean_roc_are_report_ready():
    X, y = _separable(n=30)
    res = cv_auc(X, y, n_repeats=1, n_boot=100)
    row = res.to_row(Method="ccc_rise", Rank=3)
    assert row["Method"] == "ccc_rise" and row["Rank"] == 3
    assert {"AUC", "AUCLo", "AUCHi", "Accuracy", "NComponents"} <= set(row)
    fpr, tpr = res.mean_roc()
    assert len(fpr) == len(tpr) and fpr[0] == 0.0 and tpr[-1] == 1.0
