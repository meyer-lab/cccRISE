"""Unit tests for cccrise.benchmark."""

import numpy as np
import pytest
import tensorly as tl

from cccrise.benchmark import (
    eigen_varexp_rank,
    fit_comparators,
    kneedle_elbow,
    pseudobulk_cpd_rank,
    staccato_order,
    tensor_cell2cell_elbow,
)


def _nonneg_cp_tensor(shape, rank, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    factors = [rng.uniform(size=(size, rank)) for size in shape]
    tensor = tl.cp_to_tensor((np.ones(rank), factors))
    return tensor + noise * rng.uniform(size=shape)


def test_kneedle_elbow_finds_the_bend():
    ranks = np.arange(1, 11)
    errors = [1.0, 0.5, 0.2, 0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08]
    assert kneedle_elbow(ranks, errors) == 3


def test_kneedle_elbow_short_curve_returns_minimum():
    assert kneedle_elbow([1, 2], [0.5, 0.1]) == 2


def test_eigen_varexp_rank_recovers_mode_rank():
    tensor = _nonneg_cp_tensor((12, 6, 6, 20), rank=2, noise=1e-6)
    for mode in range(4):
        rank, proportions = eigen_varexp_rank(tensor, mode, varexp=1.0)
        assert rank == 2
        assert proportions.sum() == pytest.approx(1.0)


def test_tensor_cell2cell_elbow_on_rank_three_tensor():
    tensor = _nonneg_cp_tensor((10, 5, 5, 15), rank=3, seed=3)
    rank, errors = tensor_cell2cell_elbow(tensor, upper_rank=7, runs=2)
    assert errors.shape == (7,)
    assert np.all(np.diff(errors[:3]) < 0)
    assert rank == 3


def test_pseudobulk_cpd_rank_stops_before_the_rule_fails():
    tensor = _nonneg_cp_tensor((10, 5, 5, 15), rank=2, seed=5, noise=1e-3)
    rank, diagnostics = pseudobulk_cpd_rank(tensor, max_rank=4)
    assert list(diagnostics["Rank"]) == [1, 2, 3, 4]
    assert diagnostics.loc[diagnostics["Rank"] <= rank, "CoreConsistency"].min() >= 85
    assert rank == 2


def test_staccato_order_moves_lr_to_mode_two():
    assert staccato_order(np.zeros((4, 3, 3, 9))).shape == (4, 9, 3, 3)


@pytest.mark.parametrize("rank_mode", ["own", "shared"])
def test_fit_comparators_returns_one_row_per_condition(rank_mode):
    tensor = _nonneg_cp_tensor((8, 4, 4, 12), rank=2, seed=1, noise=0.01)
    fits = fit_comparators(
        tensor, rank_mode=rank_mode, shared_rank=3, tc2c_upper_rank=5, tc2c_elbow_runs=1
    )
    assert [fit.method for fit in fits] == [
        "Pseudobulk CPD",
        "Tensor-cell2cell",
        "STACCato",
    ]
    for fit in fits:
        assert fit.rank_mode == rank_mode
        assert fit.condition_factor.shape[0] == 8
    if rank_mode == "shared":
        assert [fit.rank for fit in fits] == ["3", "3", "3x3x3x3"]
        assert all(fit.condition_factor.shape[1] == 3 for fit in fits)


def test_fit_comparators_rejects_unknown_rank_mode():
    with pytest.raises(ValueError, match="rank_mode"):
        fit_comparators(np.ones((3, 2, 2, 4)), rank_mode="best", shared_rank=2)
