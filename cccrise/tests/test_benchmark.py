"""Unit tests for cccrise.benchmark."""

import numpy as np
import pytest
import tensorly as tl

from cccrise.benchmark import (
    ComparatorFit,
    eigen_varexp_rank,
    fit_comparators,
    kneedle_elbow,
    load_fits,
    pseudobulk_cpd_rank,
    save_fits,
    staccato_order,
    tensor_cell2cell_elbow,
    transform_tensor,
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


def test_pseudobulk_cpd_rank_stopping_early_selects_the_same_rank():
    tensor = _nonneg_cp_tensor((10, 5, 5, 15), rank=2, seed=5, noise=1e-3)
    rank, full = pseudobulk_cpd_rank(tensor, max_rank=4)
    early_rank, early = pseudobulk_cpd_rank(tensor, max_rank=4, stop_at_failure=True)
    assert early_rank == rank
    assert list(early["Rank"]) == list(range(1, rank + 2))
    np.testing.assert_allclose(
        early["CoreConsistency"], full["CoreConsistency"].iloc[: rank + 1]
    )


def test_compressed_pseudobulk_cpd_matches_the_full_rank_and_span():
    from cccrise.benchmark import _pseudobulk_target, pseudobulk_cpd_fit

    tensor = _nonneg_cp_tensor((20, 12, 12, 30), rank=2, seed=3, noise=1e-3)
    core, bases, _ = _pseudobulk_target(tensor, True, 3, None, "numpy")
    assert core.shape == (8, 8, 8, 8)
    full_rank, _ = pseudobulk_cpd_rank(tensor, max_rank=3)
    core_rank, _ = pseudobulk_cpd_rank(core, max_rank=3)
    assert core_rank == full_rank == 2
    _, full = pseudobulk_cpd_fit(tensor, 2)
    _, small = pseudobulk_cpd_fit(core, 2)
    lr = bases[3] @ small[3]
    cos = np.abs(
        (lr / np.linalg.norm(lr, axis=0)).T
        @ (full[3] / np.linalg.norm(full[3], axis=0))
    )
    assert cos.max(axis=1).min() > 0.999


def test_pseudobulk_cpd_rank_never_exceeds_the_smallest_mode():
    tensor = _nonneg_cp_tensor((6, 3, 3, 10), rank=2, seed=2, noise=0.01)
    _rank, diagnostics = pseudobulk_cpd_rank(tensor, max_rank=10)
    assert diagnostics["Rank"].max() == 3


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
        assert all(set(fit.modes) == {"sender", "receiver", "lr"} for fit in fits)
        assert all(fit.modes["lr"].shape[0] == 12 for fit in fits)
        assert all(fit.modes["sender"].shape[0] == 4 for fit in fits)


def test_fit_comparators_resumes_from_its_checkpoints(tmp_path):
    tensor = _nonneg_cp_tensor((8, 4, 4, 12), rank=2, seed=1, noise=0.01)
    kwargs = {"rank_mode": "own", "shared_rank": 3, "tc2c_upper_rank": 5}
    first = fit_comparators(
        tensor, tc2c_elbow_runs=1, checkpoint_dir=tmp_path, **kwargs
    )
    assert (tmp_path / "tensor_cell2cell_elbow.pkl").exists()
    # A different tensor would refit; reading the checkpoints gives the first fits.
    again = fit_comparators(
        tensor * 2, tc2c_elbow_runs=1, checkpoint_dir=tmp_path, **kwargs
    )
    for a, b in zip(first[:2], again[:2], strict=True):
        assert a.rank == b.rank
        np.testing.assert_array_equal(a.condition_factor, b.condition_factor)


def test_transform_tensor_tempers_large_entries():
    rng = np.random.default_rng(0)
    tensor = rng.uniform(size=(6, 3, 3, 4))
    tensor[0, 0, 0] = 1000.0  # one condition x cell-type pair dominates
    winsor = transform_tensor(tensor, "winsor")
    assert winsor.max() < 1000 and np.all(winsor <= tensor)
    block = transform_tensor(tensor, "blocknorm")
    np.testing.assert_allclose(np.linalg.norm(block, axis=-1), 1.0)
    assert transform_tensor(tensor, "none") is tensor
    with pytest.raises(ValueError, match="transform"):
        transform_tensor(tensor, "log")


def test_fit_comparators_rejects_unknown_rank_mode():
    with pytest.raises(ValueError, match="rank_mode"):
        fit_comparators(np.ones((3, 2, 2, 4)), rank_mode="best", shared_rank=2)


def test_fit_cache_round_trips_and_refuses_stale_settings(tmp_path):
    conditions = np.array(["c1", "c2", "c3"])
    fits = [
        ComparatorFit(
            "Tensor-cell2cell",
            "own",
            "2",
            np.arange(6.0).reshape(3, 2),
            {"elbow_errors": [0.5, 0.2]},
            modes={
                "sender": np.ones((2, 2)),
                "receiver": np.full((2, 2), 2.0),
                "lr": np.arange(8.0).reshape(4, 2),
            },
            weights=np.array([1.5, 0.5]),
        ),
        ComparatorFit(
            "STACCato",
            "own",
            "3x2x2x2",
            np.ones((3, 3)),
            {"core_shape": [3, 2, 2, 2]},
            modes={
                "sender": np.ones((2, 2)),
                "receiver": np.ones((2, 2)),
                "lr": np.ones((4, 2)),
            },
            core=np.arange(24.0).reshape(3, 2, 2, 2),
        ),
    ]
    info = {"tensor_shape": [3, 2, 2, 4], "celltypes": ["a", "b"], "n_lr_pairs": 4}
    path = tmp_path / "fits.nc"
    save_fits(path, fits, conditions, {"seed": 1}, info)

    loaded = load_fits(path, conditions, {"seed": 1})
    assert loaded is not None
    loaded_fits, loaded_info = loaded
    assert loaded_info == info
    for original, restored in zip(fits, loaded_fits):
        assert (restored.method, restored.rank_mode, restored.rank) == (
            original.method,
            original.rank_mode,
            original.rank,
        )
        np.testing.assert_array_equal(
            restored.condition_factor, original.condition_factor
        )
        assert restored.details == original.details
        assert restored.kind == original.kind
        for mode, values in original.modes.items():
            np.testing.assert_array_equal(restored.modes[mode], values)
        for attr in ("weights", "core"):
            if getattr(original, attr) is None:
                assert getattr(restored, attr) is None
            else:
                np.testing.assert_array_equal(
                    getattr(restored, attr), getattr(original, attr)
                )

    assert load_fits(path, conditions, {"seed": 2}) is None
    assert load_fits(path, conditions[::-1], {"seed": 1}) is None
    assert load_fits(tmp_path / "missing.nc", conditions, {"seed": 1}) is None


def test_tensor_cell2cell_gpu_matches_cpu():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no GPU")
    from cccrise.benchmark import _nonneg_cp

    tensor = _nonneg_cp_tensor((6, 3, 3, 10), rank=2, seed=3, noise=0.01)
    cpu = _nonneg_cp(tensor, 2, 0)
    gpu = _nonneg_cp(tensor, 2, 0, backend="cupy")
    for a, b in zip(cpu.factors, gpu.factors, strict=True):
        assert isinstance(b, np.ndarray)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_tensor_cell2cell_elbow_gpu_matches_cpu():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no GPU")
    from cccrise.benchmark import tensor_cell2cell_elbow

    tensor = _nonneg_cp_tensor((6, 3, 3, 10), rank=2, seed=3, noise=0.01)
    cpu_rank, cpu = tensor_cell2cell_elbow(tensor, upper_rank=5, runs=2)
    gpu_rank, gpu = tensor_cell2cell_elbow(tensor, upper_rank=5, runs=2, backend="cupy")
    assert gpu_rank == cpu_rank
    np.testing.assert_allclose(gpu, cpu, rtol=1e-5, atol=1e-7)


def test_staccato_fit_selects_eigen_ranks_and_matches_svd_init():
    from cccrise.benchmark import staccato_fit
    from cccrise.comparators.staccato import staccato

    tensor = staccato_order(_nonneg_cp_tensor((12, 6, 6, 20), rank=2, noise=1e-3))
    shape, proportions, core, factors = staccato_fit(tensor)
    assert shape == tuple(eigen_varexp_rank(tensor, m)[0] for m in range(4))
    for m in range(4):
        np.testing.assert_allclose(proportions[m], eigen_varexp_rank(tensor, m)[1])
    # the Gram-eigenvector start is tucker's SVD start
    reference = staccato(tensor, shape, random_state=42)
    for mine, ref in zip(
        factors, (reference.W1, reference.W3, reference.W4, reference.W2), strict=True
    ):
        np.testing.assert_allclose(mine, ref, atol=1e-6)
    np.testing.assert_allclose(core, reference.G, atol=1e-6)


def test_staccato_fit_gpu_matches_cpu():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no GPU")
    from cccrise.benchmark import staccato_fit

    tensor = staccato_order(_nonneg_cp_tensor((12, 6, 6, 20), rank=3, noise=1e-2))
    cpu = staccato_fit(tensor)
    gpu = staccato_fit(tensor, backend="cupy")
    assert gpu[0] == cpu[0]
    for a, b in zip(cpu[3], gpu[3], strict=True):
        assert isinstance(b, np.ndarray)
        np.testing.assert_allclose(a, b, atol=1e-8)
    np.testing.assert_allclose(cpu[2], gpu[2], atol=1e-8)


def test_tensor_cell2cell_elbow_resumes_from_its_progress(tmp_path, monkeypatch):
    from cccrise import benchmark

    tensor = _nonneg_cp_tensor((10, 5, 5, 15), rank=3, seed=3)
    progress = tmp_path / "elbow.json"
    full = tensor_cell2cell_elbow(tensor, upper_rank=5, runs=2, progress=progress)
    # every rank is read back: no fit runs again
    monkeypatch.setattr(benchmark, "_nonneg_cp", lambda *a, **k: 1 / 0)
    again = tensor_cell2cell_elbow(tensor, upper_rank=5, runs=2, progress=progress)
    assert again[0] == full[0]
    np.testing.assert_array_equal(again[1], full[1])


@pytest.mark.parametrize("shape", [(4, 3, 3, 6), (5, 3, 3, 7)])  # even, odd inner sizes
def test_odd_inner_mttkrp_matches_tensorly(shape):
    from cccrise.benchmark import _odd_inner_mttkrp

    rng = np.random.default_rng(0)
    tensor = rng.random(shape)
    cp = (rng.random(2), [rng.random((n, 2)) for n in shape])
    for mode in range(4):
        np.testing.assert_allclose(
            _odd_inner_mttkrp(tensor, cp, mode),
            tl.tenalg.unfolding_dot_khatri_rao(tensor, cp, mode),
            rtol=1e-12,
        )


def test_odd_inner_mttkrp_leaves_float32_unpadded():
    from cccrise.benchmark import _odd_inner_mttkrp

    rng = np.random.default_rng(0)
    tensor = rng.random((4, 3, 3, 6)).astype(np.float32)
    cp = (
        rng.random(2).astype(np.float32),
        [rng.random((n, 2)).astype(np.float32) for n in tensor.shape],
    )
    for mode in range(4):
        np.testing.assert_array_equal(
            _odd_inner_mttkrp(tensor, cp, mode),
            tl.tenalg.unfolding_dot_khatri_rao(tensor, cp, mode),
        )
