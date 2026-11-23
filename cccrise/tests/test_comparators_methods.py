"""Unit tests for cccrise.comparators.tensors and cccrise.comparators.methods."""

from typing import Any

import anndata
import numpy as np
import pandas as pd
import pytest
import tensorly as tl
from scipy import sparse
from tensorly.decomposition import non_negative_parafac

from cccrise.comparators import methods
from cccrise.comparators.tensors import (
    PooledTensor,
    pseudobulk_tensor,
    staccato_tensor,
    tensor_cell2cell_tensor,
)
from cccrise.rank_selection import kneedle_elbow

FIT: dict[str, Any] = {"n_iter_max": 100, "tol": 1e-7}
LR = pd.DataFrame(
    {
        "ligand": ["L1", "L2", "L1"],
        "receptor": ["R1&R2", "R1", "MISSING"],
        "interaction_symbol": ["L1^R1&R2", "L2^R1", "L1^MISSING"],
    }
)


@pytest.fixture
def adata():
    """Two samples; type "b" has no cells in sample "s2"."""
    X = np.array(
        [
            # L1, L2, R1, R2
            [2.0, 0.0, 1.0, 4.0],  # s1 a
            [0.0, 1.0, 3.0, 0.0],  # s1 a
            [1.0, 1.0, 0.0, 2.0],  # s1 b
            [4.0, 0.0, 2.0, 2.0],  # s2 a
        ]
    )
    obs = pd.DataFrame(
        {"sample": ["s1", "s1", "s1", "s2"], "type": ["a", "a", "b", "a"]}
    )
    data = anndata.AnnData(sparse.csr_matrix(X), obs=obs)
    data.var_names = ["L1", "L2", "R1", "R2"]
    return data


def test_pseudobulk_tensor_scores_mean_products(adata):
    pooled = pseudobulk_tensor(adata, "sample", "type", LR)
    assert pooled.lr_pairs.tolist() == ["L1^R1&R2", "L2^R1"]
    assert pooled.mask is None
    # s1, a -> b: mean L1 over a's cells (1.0) x min(R1, R2) of b (min(0, 2) = 0)
    assert pooled.values[0, 0, 1, 0] == 0.0
    # s1, b -> a: L2 of b (1.0) x R1 mean of a (2.0)
    assert pooled.values[0, 1, 0, 1] == pytest.approx(2.0)
    # s2 has no b cells: zero, not masked
    assert np.all(pooled.values[1, 1] == 0) and np.all(pooled.values[1, :, 1] == 0)


def test_tensor_cell2cell_tensor_masks_absent_types(adata):
    pooled = tensor_cell2cell_tensor(adata, "sample", "type", LR)
    # s1, a -> a, L2^R1: fraction expressing L2 in a (0.5) and R1 (1.0), averaged
    assert pooled.values[0, 0, 0, 1] == pytest.approx(0.75)
    # s1, a -> a, L1^R1&R2: L1 fraction 0.5, complex min(1.0, 0.5) = 0.5
    assert pooled.values[0, 0, 0, 0] == pytest.approx(0.5)
    mask = pooled.mask
    assert mask is not None
    assert np.all(mask[0] == 1)
    assert np.all(mask[1, 1] == 0) and np.all(mask[1, :, 1] == 0)
    assert mask[1, 0, 0].all() and np.all(pooled.values[1, 1] == 0)


def test_masked_error_ignores_masked_entries():
    rng = np.random.default_rng(0)
    factors = [rng.random((n, 2)) for n in (3, 4, 4, 5)]
    tensor = tl.cp_to_tensor((None, factors))
    mask = np.ones_like(tensor)
    mask[0, 0] = 0
    noisy = tensor.copy()
    noisy[0, 0] += 100.0
    assert methods.masked_error(noisy, mask, np.ones(2), factors) == pytest.approx(0)
    assert methods.masked_error(noisy, None, np.ones(2), factors) > 0.1


def test_tensor_cell2cell_fit_is_upstream_mode_order():
    """A seed's fit is the fit of the tensor in Tensor-cell2cell's
    (sample, lr, sender, receiver) order, whose random start it shares."""
    rng = np.random.default_rng(1)
    tensor = rng.random((3, 4, 4, 5))
    _, factors = methods.tensor_cell2cell_fit(
        tensor, None, 2, seed=7, device="cpu", dtype="float64", **FIT
    )
    upstream = non_negative_parafac(
        np.transpose(tensor, (0, 3, 1, 2)),
        2,
        init="random",
        random_state=7,
        **FIT,
    )
    sample, lr, sender, receiver = upstream.factors
    for ours, theirs in zip(factors, (sample, sender, receiver, lr), strict=True):
        np.testing.assert_allclose(ours, theirs)


def test_staccato_core_shape_counts_shares_above_threshold():
    shares = pd.DataFrame(
        {
            "Mode": ["sample"] * 3 + ["sender", "receiver", "lr"],
            "Component": [1, 2, 3, 1, 1, 1],
            "Share": [0.9, 0.05, 0.005, 0.001, 1.0, 1.0],
        }
    )
    assert methods.staccato_core_shape(shares) == {
        "sample": 2,
        "sender": 1,
        "receiver": 1,
        "lr": 1,
    }


def test_kneedle_elbow_finds_the_bend():
    ranks = np.arange(1, 9)
    assert kneedle_elbow(ranks, [1.0, 0.5, 0.25, 0.2, 0.18, 0.17, 0.16, 0.15]) == 3


def test_tensor_cell2cell_elbow_keeps_each_runs_factors():
    rng = np.random.default_rng(2)
    tensor = rng.random((3, 4, 4, 5))
    elbow, factors = methods.tensor_cell2cell_elbow(
        tensor, None, [1, 2], runs=3, device="cpu", **FIT
    )
    assert list(factors) == [1, 2] and list(factors[2]) == [42, 43, 44]
    np.testing.assert_allclose(elbow["FitR2X"], 1 - elbow["Error"] ** 2)
    pooled = PooledTensor(
        tensor, None, *(np.array(list("abcde"[:n])) for n in (3, 4, 5))
    )
    starts = methods.starts_dataset(pooled, factors)
    rank_2 = starts.sel(start_component=(starts["rank"] == 2) & (starts["seed"] == 43))
    np.testing.assert_array_equal(rank_2["lr_factor"], factors[2][43][3])
    read_back = methods.starts_factors(starts)
    for rank, by_seed in factors.items():
        for seed, expected in by_seed.items():
            for ours, theirs in zip(read_back[rank][seed], expected, strict=True):
                np.testing.assert_array_equal(ours, theirs)


def test_tensor_cell2cell_inner_keeps_only_types_in_every_sample(adata):
    pooled = tensor_cell2cell_tensor(adata, "sample", "type", LR, how="inner")
    assert pooled.celltypes.tolist() == ["a"]
    assert pooled.mask is None
    assert pooled.values.shape == (2, 1, 1, 2)


def test_best_start_is_the_lowest_error_run():
    rng = np.random.default_rng(4)
    tensor = rng.random((3, 4, 4, 5))
    elbow, factors = methods.tensor_cell2cell_elbow(
        tensor, None, [2], runs=3, device="cpu", **FIT
    )
    seed, weights, best = methods.best_start(elbow, factors, 2)
    assert seed == elbow.loc[elbow["Error"].idxmin(), "Seed"]
    refit_weights, refit = methods.tensor_cell2cell_fit(
        tensor, None, 2, seed, device="cpu", **FIT
    )
    np.testing.assert_array_equal(weights, refit_weights)
    for ours, theirs in zip(best, refit, strict=True):
        np.testing.assert_allclose(ours, theirs)


def test_tensor_cell2cell_svd_start_ignores_the_seed():
    rng = np.random.default_rng(5)
    tensor = rng.random((3, 4, 4, 5))
    fits = [
        methods.tensor_cell2cell_fit(
            tensor, None, 2, s, device="cpu", init="svd", **FIT
        )
        for s in (1, 2)
    ]
    for a, b in zip(fits[0][1], fits[1][1], strict=True):
        np.testing.assert_allclose(a, b)


def test_tensor_cell2cell_elbow_workers_do_not_change_the_fits():
    rng = np.random.default_rng(6)
    tensor = rng.random((3, 4, 4, 5))
    serial, serial_factors = methods.tensor_cell2cell_elbow(
        tensor, None, [1, 2], runs=2, device="cpu", **FIT
    )
    parallel, parallel_factors = methods.tensor_cell2cell_elbow(
        tensor, None, [1, 2], runs=2, device="cpu", workers=2, **FIT
    )
    pd.testing.assert_frame_equal(serial, parallel)
    for rank, by_seed in serial_factors.items():
        for seed, factors in by_seed.items():
            for ours, theirs in zip(factors, parallel_factors[rank][seed], strict=True):
                np.testing.assert_array_equal(ours, theirs)


def test_staccato_tensor_applies_lianas_filters(adata):
    lenient = staccato_tensor(adata, "sample", "type", LR, expr_prop=0.0, min_cells=1)
    # s1, a -> a, L2^R1: mean L2 in a (0.5) x mean R1 in a (2.0), as PB scores it
    assert lenient.values[0, 0, 0, 1] == pytest.approx(1.0)
    assert lenient.mask is None
    # L2 is detected in half of a's cells: below a 0.6 threshold, the pair scores 0
    strict = staccato_tensor(adata, "sample", "type", LR, expr_prop=0.6, min_cells=1)
    assert strict.values[0, 0, 0, 1] == 0.0
    # b has one cell in s1: under min_cells=2 it counts as absent there
    few = staccato_tensor(adata, "sample", "type", LR, expr_prop=0.0, min_cells=2)
    assert np.all(few.values[0, 1] == 0) and np.all(few.values[0, :, 1] == 0)
    assert few.values[0, 0, 0, 1] == pytest.approx(1.0)


def test_fmax_best_seed_is_the_best_start_of_the_most_reached_minimum(tmp_path):
    from cccrise.runners.run_comparators import fmax_best_seed

    # Seeds 1-3 reach one minimum, seed 4 (the best fit) another
    r2x = pd.DataFrame(
        {"Rank": 2, "Seed": [1, 2, 3, 4], "FitR2X": [0.80, 0.81, 0.79, 0.90]}
    )
    fms = pd.DataFrame(
        [
            {"Rank": 2, "SeedA": a, "SeedB": b, "FMS": 1.0 if b < 4 else 0.5}
            for a in range(1, 5)
            for b in range(a + 1, 5)
        ]
    )
    sweep = {"rank": tmp_path / "rank.parquet", "start_fms": tmp_path / "fms.parquet"}
    r2x.to_parquet(sweep["rank"])
    fms.to_parquet(sweep["start_fms"])
    assert fmax_best_seed(sweep, 2) == 2
