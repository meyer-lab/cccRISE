"""Tests for cellcommunicationpf2.rank_selection."""

from itertools import pairwise

import anndata
import numpy as np
import pandas as pd
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from cellcommunicationpf2.rank_selection import (
    bicv_reconstruction_error,
    cp_core_consistency,
    cp_core_consistency_sweep,
    cp_fit_r2x_sweep,
    cpd_component_collinearity,
    run_bicv,
)


def _simulate_parafac2(
    n_cond: int,
    cells_per_cond: int,
    n_genes: int,
    rank: int,
    noise_std: float = 0.05,
    random_state: int = 0,
) -> anndata.AnnData:
    """Simulate data X_i = P_i diag(a_i) B C^T + noise from a known PARAFAC2 model."""
    rng = np.random.default_rng(random_state)
    A = rng.uniform(0.5, 1.5, size=(n_cond, rank))
    B = rng.normal(size=(rank, rank))
    C = rng.normal(size=(n_genes, rank))

    blocks = []
    cond_ids = []
    for i in range(n_cond):
        M = rng.normal(size=(cells_per_cond, rank))
        P_i, _ = np.linalg.qr(M)
        X_i = (P_i * A[i]) @ B @ C.T
        X_i += rng.normal(scale=noise_std, size=X_i.shape)
        blocks.append(X_i)
        cond_ids.extend([f"cond{i}"] * cells_per_cond)

    X = np.concatenate(blocks, axis=0).astype(np.float32)
    obs = pd.DataFrame({"sample": cond_ids})
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.var["means"] = X.mean(axis=0)
    return adata


def test_bicv_reconstruction_error_rises_past_true_rank():
    """Verify that held-out BiCV R2X peaks/plateaus near the true rank."""
    true_rank = 3
    adata = _simulate_parafac2(
        n_cond=6,
        cells_per_cond=60,
        n_genes=40,
        rank=true_rank,
        noise_std=0.02,
        random_state=42,
    )

    results = {}
    for rank in (1, 2, 3, 6, 10):
        res = bicv_reconstruction_error(
            adata,
            rank=rank,
            condition_column="sample",
            gene_test_frac=0.25,
            n_iter_max=200,
            tol=1e-8,
            random_state=0,
        )
        results[rank] = res

    # In-sample training-block R2X must climb monotonically with rank
    train_r2x = [results[r]["TrainBlockR2X"] for r in (1, 2, 3, 6, 10)]
    assert all(x <= y + 1e-6 for x, y in pairwise(train_r2x))

    # The held-out block R2X at the true rank should be substantially better than at rank 1
    assert results[true_rank]["HeldOutBlockR2X"] > results[1]["HeldOutBlockR2X"]

    # And it should not be monotonically increasing all the way to rank 10
    held_out = [results[r]["HeldOutBlockR2X"] for r in (1, 2, 3, 6, 10)]
    assert not all(x <= y + 1e-6 for x, y in pairwise(held_out))


def test_bicv_train_block_r2x_matches_parafac2_report():
    """Verify bounds and valid R2X output."""
    adata = _simulate_parafac2(
        n_cond=4, cells_per_cond=30, n_genes=20, rank=2, random_state=7
    )
    res = bicv_reconstruction_error(
        adata,
        rank=2,
        condition_column="sample",
        gene_test_frac=0.2,
        n_iter_max=100,
        tol=1e-8,
        random_state=1,
    )
    assert 0.0 <= res["TrainBlockR2X"] <= 1.0
    assert -1.0 <= res["HeldOutBlockR2X"] <= 1.0


def test_run_bicv_interface():
    """Verify modular run_bicv interface across multiple ranks and folds."""
    adata = _simulate_parafac2(
        n_cond=3, cells_per_cond=20, n_genes=15, rank=2, random_state=8
    )
    ranks = [1, 2]
    n_folds = 3
    df = run_bicv(
        adata,
        ranks=ranks,
        condition_column="sample",
        n_folds=n_folds,
        n_iter_max=50,
        tol=1e-6,
    )

    assert len(df) == len(ranks) * n_folds
    assert set(df["Rank"].unique()) == set(ranks)
    assert set(df["Fold"].unique()) == set(range(n_folds))
    assert "HeldOutBlockR2X" in df.columns
    assert "TrainBlockR2X" in df.columns


# =============================================================================
# CPD (interaction-tensor) rank-selection tests
# =============================================================================


def _simulate_cp_tensor(
    shape: tuple[int, ...], rank: int, noise_std: float = 0.05, random_state: int = 0
) -> np.ndarray:
    """Simulate a plain CP tensor of a known true rank, plus noise."""
    rng = np.random.default_rng(random_state)
    factors = [rng.normal(size=(dim, rank)) for dim in shape]
    weights = np.ones(rank)
    tensor = cp_to_tensor((weights, factors))
    tensor += rng.normal(scale=noise_std, size=tensor.shape)
    return tensor


def test_cp_fit_r2x_sweep_monotonic_and_bounded():
    """Full-tensor fit R2X (nothing held out) must rise monotonically with rank."""
    tensor = _simulate_cp_tensor((6, 8, 8, 20), rank=3, random_state=3)
    ranks = [1, 2, 3, 6]
    df = cp_fit_r2x_sweep(tensor, ranks=ranks, n_iter_max=200, random_state=0)

    assert list(df["Rank"]) == ranks
    assert df["FitR2X"].between(0.0, 1.0).all()
    assert all(x <= y + 1e-6 for x, y in pairwise(df["FitR2X"]))


def _factors_with_pair(
    dims: tuple[int, ...], flip_mode: int | None
) -> list[np.ndarray]:
    """Rank-3 factor matrices where components 0 and 1 are near-duplicates in
    every mode, negated in `flip_mode` if given (the canceling-pair signature).
    Component 2 is independent random noise."""
    rng = np.random.default_rng(0)
    factors = []
    for mode, dim in enumerate(dims):
        base = rng.normal(size=dim)
        col1 = -base if mode == flip_mode else base
        other = rng.normal(size=(dim, 1))
        factors.append(np.column_stack([base, col1, other.ravel()]))
    return factors


def test_cpd_component_collinearity_flags_duplicate_and_canceling_pairs():
    weights = np.ones(3)
    dims = (5, 6, 6, 8)

    duplicate = {3: (weights, _factors_with_pair(dims, flip_mode=None))}
    df = cpd_component_collinearity(duplicate)
    assert df.loc[df["Rank"] == 3, "MaxModesCollinear"].item() == 4
    assert df.loc[df["Rank"] == 3, "WorstPairCanceling"].item() == False

    canceling = {3: (weights, _factors_with_pair(dims, flip_mode=1))}
    df = cpd_component_collinearity(canceling)
    assert df.loc[df["Rank"] == 3, "WorstPairCanceling"].item() == True


def test_cpd_component_collinearity_rank_below_two_is_trivial():
    df = cpd_component_collinearity({1: (np.ones(1), [np.ones((4, 1))] * 4)})
    row = df.iloc[0]
    assert row["MaxModesCollinear"] == 0
    assert pd.isna(row["WorstPairCanceling"])


def test_cp_core_consistency_drops_past_true_rank():
    """CORCONDIA should be high at the true rank and lower once overfit."""
    true_rank = 3
    tensor = _simulate_cp_tensor(
        (10, 12, 12, 30), rank=true_rank, noise_std=0.02, random_state=5
    )

    def _consistency(rank: int) -> float:
        weights, factors = parafac(
            tensor,
            rank=rank,
            n_iter_max=300,
            init="svd",
            normalize_factors=True,
            random_state=0,
        )
        return cp_core_consistency(tensor, weights, factors)

    at_true_rank = _consistency(true_rank)
    overfit = _consistency(true_rank + 5)
    assert at_true_rank > 50.0
    assert overfit < at_true_rank


def test_cp_core_consistency_sweep_shape():
    tensor = _simulate_cp_tensor((6, 7, 7, 15), rank=2, random_state=6)
    factors_by_rank = {
        rank: parafac(
            tensor, rank=rank, n_iter_max=100, normalize_factors=True, random_state=0
        )
        for rank in (1, 2, 4)
    }
    df = cp_core_consistency_sweep(tensor, factors_by_rank)
    assert set(df["Rank"]) == {1, 2, 4}
    assert "CoreConsistency" in df.columns
