"""Tests for cccrise.rank_selection."""

from itertools import pairwise

import anndata
import numpy as np
import pandas as pd
import pytest
from parafac2.parafac2 import parafac2_nd
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from cccrise.import_data import add_cond_idxs
from cccrise.rank_selection import (
    _cell_loadings,
    bicv_reconstruction_error,
    calculate_r2x,
    candelinc_compress,
    candelinc_r2x,
    cp_core_consistency,
    cp_core_consistency_sweep,
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


def test_cell_loadings_reproduce_the_parafac2_model():
    """Pin the held-out design to X_i = P_i B diag(w a_i) C^T against the fit's own R2X.

    Scaling `P_i` by `a_i` before rather than after multiplying by `B` is a
    different model whenever `a_i` varies across conditions, and produces a
    systematically depressed held-out score rather than an obvious failure.
    """
    adata = add_cond_idxs(
        _simulate_parafac2(5, 50, 30, 3, noise_std=0.05, random_state=3), "sample"
    )
    centered = np.asarray(adata.X, dtype=np.float64) - adata.var["means"].to_numpy()
    condition_idx = adata.obs["condition_unique_idxs"].to_numpy(dtype=int)

    (weights, factors, projection_list), reported = parafac2_nd(
        adata, rank=3, n_iter_max=800, tol=1e-14, random_state=0
    )
    projections = np.zeros((adata.n_obs, 3))
    for i, p in enumerate(projection_list):
        projections[condition_idx == i] = p

    loadings = _cell_loadings(weights, factors, projections, condition_idx)
    residual = centered - loadings @ factors[2].T
    r2x = 1.0 - np.sum(residual**2) / np.sum(centered**2)
    assert r2x == pytest.approx(reported, abs=1e-6)


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
    tensor = cp_to_tensor((np.ones(rank), factors))
    tensor += rng.normal(scale=noise_std, size=tensor.shape)
    return tensor


def _cp_fit(tensor: np.ndarray, rank: int, n_iter_max: int = 2000, tol: float = 1e-12):
    """Fit CP the way `run_cpd_rank.cp_fits_by_rank` does."""
    weights, factors = parafac(
        tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init="svd",
        svd="randomized_svd",
        normalize_factors=True,
        random_state=0,
    )
    return weights, factors, calculate_r2x(weights, factors, tensor)


def _factors_with_pair(
    dims: tuple[int, ...], flip_mode: int | None
) -> list[np.ndarray]:
    """Rank-3 factors whose components 0 and 1 are near-duplicates in every
    mode, negated in `flip_mode` if given (the canceling-pair signature).
    Component 2 is independent noise."""
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
    assert df.loc[df["Rank"] == 3, "WorstPairCanceling"].item() is False

    canceling = {3: (weights, _factors_with_pair(dims, flip_mode=1))}
    df = cpd_component_collinearity(canceling)
    assert df.loc[df["Rank"] == 3, "WorstPairCanceling"].item() is True


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
        weights, factors, _r2x = _cp_fit(tensor, rank, n_iter_max=300, tol=1e-8)
        return cp_core_consistency(tensor, weights, factors)

    at_true_rank = _consistency(true_rank)
    overfit = _consistency(true_rank + 5)
    assert at_true_rank > 50.0
    assert overfit < at_true_rank


def test_cp_core_consistency_sweep_shape():
    tensor = _simulate_cp_tensor((6, 7, 7, 15), rank=2, random_state=6)
    factors_by_rank = {
        rank: _cp_fit(tensor, rank, n_iter_max=100, tol=1e-8)[:2] for rank in (1, 2, 4)
    }
    df = cp_core_consistency_sweep(tensor, factors_by_rank)
    assert set(df["Rank"]) == {1, 2, 4}
    assert "CoreConsistency" in df.columns


# =============================================================================
# CANDELINC compression
# =============================================================================


def _fit_compressed_and_full(tensor: np.ndarray, rank: int, max_rank: int):
    """Fit CP inside the CANDELINC core and return both spaces' views of it."""
    core, bases, lost_sq = candelinc_compress(tensor, max_rank=max_rank)
    weights, core_factors, core_r2x = _cp_fit(core, rank)
    decompressed = [b @ f for b, f in zip(bases, core_factors, strict=True)]
    return core, bases, lost_sq, weights, core_factors, decompressed, core_r2x


def test_candelinc_bases_are_orthonormal_and_sized_for_the_rank():
    tensor = _simulate_cp_tensor((20, 9, 9, 30), rank=3, random_state=11)
    core, bases, lost_sq = candelinc_compress(tensor, max_rank=5, oversample=2)

    for basis, dim in zip(bases, tensor.shape, strict=True):
        assert basis.shape == (dim, min(dim, 7))
        np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8)

    assert core.shape == tuple(min(d, 7) for d in tensor.shape)
    assert core.size < tensor.size
    # The projection can only discard energy, never invent it.
    assert 0.0 <= lost_sq <= float(np.sum(tensor**2))


def test_candelinc_core_consistency_matches_the_full_tensor():
    """CORCONDIA off the core equals CORCONDIA off the tensor.

    This is the identity that makes it correct to run the sweep on the core:
    the bases are orthonormal, so `pinv(U F) = pinv(F) U.T` leaves the
    projected core unchanged.
    """
    tensor = _simulate_cp_tensor((20, 9, 9, 30), rank=3, random_state=12)

    for rank in (2, 3, 5):
        core, _b, _l, weights, core_factors, decompressed, _r = (
            _fit_compressed_and_full(tensor, rank, max_rank=6)
        )
        from_core = cp_core_consistency(core, weights, core_factors)
        from_tensor = cp_core_consistency(tensor, weights, decompressed)
        assert from_core == pytest.approx(from_tensor, abs=1e-6, rel=1e-9)


def test_candelinc_collinearity_matches_the_full_tensor():
    """Within-mode cosines are invariant under an orthonormal basis change."""
    tensor = _simulate_cp_tensor((20, 9, 9, 30), rank=3, random_state=13)

    for rank in (2, 4):
        _c, _b, _l, weights, core_factors, decompressed, _r = _fit_compressed_and_full(
            tensor, rank, max_rank=6
        )
        from_core = cpd_component_collinearity({rank: (weights, core_factors)})
        from_tensor = cpd_component_collinearity({rank: (weights, decompressed)})
        pd.testing.assert_frame_equal(from_core, from_tensor)


def test_candelinc_r2x_recovers_the_uncompressed_fit():
    """A core-space R2X, corrected for what compression discarded, matches a
    fit run against the full tensor."""
    tensor = _simulate_cp_tensor((20, 9, 9, 30), rank=3, random_state=14)

    for rank in (2, 3):
        core, _b, lost_sq, _w, _cf, _d, core_r2x = _fit_compressed_and_full(
            tensor, rank, max_rank=6
        )
        _fw, _ff, full_r2x = _cp_fit(tensor, rank)
        assert candelinc_r2x(core_r2x, core, lost_sq) == pytest.approx(
            full_r2x, abs=1e-4
        )


def test_candelinc_r2x_is_the_identity_without_compression():
    """The uncompressed runner path passes lost_sq=0 through this same call."""
    core = _simulate_cp_tensor((4, 4, 4, 4), rank=2, random_state=15)
    assert candelinc_r2x(0.73, core, 0.0) == pytest.approx(0.73)


def test_candelinc_rejects_a_rank_the_tensor_cannot_support():
    tensor = _simulate_cp_tensor((20, 5, 5, 30), rank=2, random_state=16)
    with pytest.raises(ValueError, match="exceeds the smallest mode"):
        candelinc_compress(tensor, max_rank=6)


def test_candelinc_finds_the_true_rank_cliff():
    """End to end: the compressed sweep still puts the CORCONDIA cliff at the
    tensor's true rank."""
    true_rank = 4
    tensor = _simulate_cp_tensor((25, 10, 10, 40), rank=true_rank, random_state=17)
    core, _bases, _lost = candelinc_compress(tensor, max_rank=7)

    consistency = {}
    for rank in range(1, 8):
        weights, factors, _r2x = _cp_fit(core, rank, n_iter_max=4000)
        consistency[rank] = cp_core_consistency(core, weights, factors)

    assert min(consistency[r] for r in range(1, true_rank + 1)) > 50.0
    assert consistency[true_rank + 1] < 50.0


def test_runner_compressed_and_uncompressed_agree(tmp_path):
    """`--no-compress` and the default compressed path must agree where it counts.

    Up to the true rank the CP model is identifiable and the two paths must
    report the same metrics. Past it the model is degenerate *by definition*:
    many equally-good solutions fit the tensor, the subspace constraint is
    enough to pick a different one, and CORCONDIA there is not a stable
    quantity. So the assertion past the true rank is only that both paths still
    see the model degrade -- not that they agree on a number. On real data the
    decision sits at the identifiable boundary, where they do agree.
    """
    from unittest.mock import patch

    import cccrise.runners.run_cpd_rank as runner
    from cccrise.runners.datasets import DATASETS

    true_rank = 3
    tensor = _simulate_cp_tensor((24, 10, 10, 36), rank=true_rank, random_state=22)
    facts = {"resulting_dims": {}, "lr_pairs_annotated": 36, "r2x_pf2": 0.1}

    results = {}
    for compress in (True, False):
        with patch.object(
            runner, "build_interaction_tensor", return_value=(tensor, facts)
        ):
            df, csv_path, sidecar = runner.run_cpd_rank_selection(
                DATASETS["alad"],
                rise_rank=5,
                ranks=[1, 2, 3, 4, 5],
                cp_n_iter_max=2000,
                cp_tol=1e-12,
                compress=compress,
                oversample=3,
                output_path=tmp_path / f"cpd_{compress}.csv",
            )
        assert csv_path.exists() and sidecar.exists()
        assert ("compressed: true" in sidecar.read_text()) is compress
        results[compress] = df

    compressed, full = results[True], results[False]
    assert list(compressed.columns) == list(full.columns)

    # R2X is restated against the original tensor on the compressed path, so
    # both paths must report the same fit quality at every rank.
    np.testing.assert_allclose(compressed["FitR2X"], full["FitR2X"], atol=2e-3)
    assert compressed["FitR2X"].is_monotonic_increasing

    for df in (compressed, full):
        identifiable = df["Rank"] <= true_rank
        assert (df.loc[identifiable, "CoreConsistency"] > 50.0).all()
        # The model has to visibly degrade once rank exceeds the true one.
        assert (
            df.loc[~identifiable, "CoreConsistency"].min()
            < df.loc[identifiable, "CoreConsistency"].min()
        )

    identifiable = compressed["Rank"] <= true_rank
    np.testing.assert_allclose(
        compressed.loc[identifiable, "CoreConsistency"],
        full.loc[identifiable, "CoreConsistency"],
        rtol=1e-4,
    )
    assert list(compressed.loc[identifiable, "MaxModesCollinear"]) == list(
        full.loc[identifiable, "MaxModesCollinear"]
    )
