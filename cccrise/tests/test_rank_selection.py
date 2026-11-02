"""Tests for cccrise.rank_selection.

Only the CPD stage is covered here. The RISE-side bi-cross-validation and its
compression moved to `scrise`, and are tested there; what remains is the
interaction tensor's own rank criteria: fit R2X and collinearity across seeded
starts, the distinct minima those starts reach, and imputation.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from cccrise.ccc_rise import CP_FIT_OPTIONS
from cccrise.rank_selection import (
    calculate_r2x,
    cp_backend,
    cp_imputation_sweep,
    cpd_component_collinearity,
    distinct_minima,
    fit_cp_on_device,
    masked_cp_reconstruction,
    max_mode_cosines,
)

IMPUTATION_FIT: dict[str, Any] = {**CP_FIT_OPTIONS, "n_iter_max": 500, "tol": 1e-10}


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
    """Fit CP with the production options, from one seeded start."""
    weights, factors = parafac(
        tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=0,
        **CP_FIT_OPTIONS,
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


def test_calculate_r2x_matches_the_reconstruction():
    tensor = _simulate_cp_tensor((5, 6, 6, 9), rank=2, noise_std=0.5, random_state=4)
    weights, factors, r2x = _cp_fit(tensor, 2, n_iter_max=100, tol=1e-8)
    residual = tensor - cp_to_tensor((weights, factors))
    assert r2x == pytest.approx(1 - np.sum(residual**2) / np.sum(tensor**2), abs=1e-10)


def test_max_mode_cosines_reads_the_most_collinear_pair_per_mode():
    factors = _factors_with_pair((5, 6, 6, 8), flip_mode=1)
    assert max_mode_cosines(factors) == pytest.approx([1.0] * 4)
    assert np.isnan(max_mode_cosines([np.ones((4, 1))] * 4)).all()


def _start_fms(rank: int, fms: dict[tuple[int, int], float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Rank": rank, "SeedA": a, "SeedB": b, "FMS": v} for (a, b), v in fms.items()
    )


def test_distinct_minima_groups_starts_by_agreement():
    """Starts 0-2 agree (0-2 only through 1), start 3 is alone: two minima,
    the larger first, each scored against the larger's best start."""
    fms = {
        (0, 1): 0.99,
        (1, 2): 0.97,
        (0, 2): 0.9,
        (0, 3): 0.6,
        (1, 3): 0.6,
        (2, 3): 0.7,
    }
    r2x = pd.Series(
        [0.5, 0.52, 0.51, 0.4],
        index=pd.MultiIndex.from_product([[4], range(4)], names=["Rank", "Seed"]),
    )
    minima = distinct_minima(_start_fms(4, fms), r2x)
    assert minima["Starts"].tolist() == [3, 1]
    assert minima["R2X"].tolist() == [0.52, 0.4]
    assert minima["FMS"].tolist() == [1.0, 0.6]
    assert minima["Order"].tolist() == [1, 2]
    assert not minima["Tied"].any()
    assert minima["Seeds"].tolist() == [(0, 1, 2), (3,)]


def test_distinct_minima_ties_have_no_fmax():
    fms = {
        (0, 1): 0.99,
        (2, 3): 0.99,
        (0, 2): 0.5,
        (0, 3): 0.5,
        (1, 2): 0.5,
        (1, 3): 0.5,
    }
    r2x = pd.Series(
        [0.4, 0.4, 0.6, 0.6],
        index=pd.MultiIndex.from_product([[4], range(4)], names=["Rank", "Seed"]),
    )
    minima = distinct_minima(_start_fms(4, fms), r2x)
    assert minima["Tied"].all()
    # The reference is the tied minimum with the higher R2X
    assert minima["R2X"].tolist() == [0.6, 0.4]
    assert minima["FMS"].tolist() == [1.0, 0.5]


# =============================================================================
# Device and precision
# =============================================================================


def test_cp_backend_picks_the_device():
    assert cp_backend("cpu") == "numpy"
    with pytest.raises(ValueError, match="device"):
        cp_backend("tpu")


def test_float32_and_float64_fits_agree():
    from cccrise.validation.stability import per_component_fms

    tensor = _simulate_cp_tensor((8, 9, 9, 20), rank=3, random_state=7)
    fits = {
        dtype: fit_cp_on_device(
            tensor, 3, "cpu", dtype, random_state=0, **IMPUTATION_FIT
        )
        for dtype in ("float32", "float64")
    }
    for weights, factors in fits.values():
        assert weights.dtype == np.float64
        assert all(f.dtype == np.float64 for f in factors)
    fms = per_component_fms(fits["float32"][1], fits["float64"][1])["FMS"]
    assert (fms > 0.999).all()


# =============================================================================
# Entry-masked imputation
# =============================================================================


def test_masked_cp_reconstruction_imputes_a_low_rank_tensor():
    """Held-out entries of a noise-free low-rank tensor are recovered, and the
    values at those entries never reach the fit."""
    tensor = _simulate_cp_tensor((8, 9, 9, 20), rank=3, noise_std=0.0, random_state=1)
    observed = np.random.default_rng(0).uniform(size=tensor.shape) > 0.2
    recon = masked_cp_reconstruction(tensor, observed, 3, **IMPUTATION_FIT)
    np.testing.assert_allclose(recon[~observed], tensor[~observed], atol=1e-4)

    corrupted = np.where(observed, tensor, 1e3)
    np.testing.assert_allclose(
        masked_cp_reconstruction(corrupted, observed, 3, **IMPUTATION_FIT), recon
    )


def test_cp_imputation_peaks_at_the_true_rank():
    true_rank = 3
    tensor = _simulate_cp_tensor(
        (10, 12, 12, 30), rank=true_rank, noise_std=1.0, random_state=5
    )
    df = cp_imputation_sweep(
        tensor, [1, 2, 3, 6], n_repeats=2, n_folds=5, l2_reg=0.1, **IMPUTATION_FIT
    )
    assert len(df) == 2 * 4 and set(df["Repeat"]) == {0, 1}
    r2x = df.groupby("Rank")["ImputationR2X"]
    assert r2x.mean().idxmax() == true_rank
    assert (r2x.std() < 0.01).all()


def test_ridge_comparison_matches_a_weak_ridge_to_the_unregularized_fit():
    from cccrise.runners.run_cpd_rank import ridge_comparison

    tensor = _simulate_cp_tensor((8, 9, 9, 20), rank=3, random_state=2)
    df, fits = ridge_comparison(tensor, [2, 3], 1e-3, device="cpu", **IMPUTATION_FIT)
    assert len(fits[3]) == 2
    assert df.groupby("Rank").size().to_dict() == {2: 2, 3: 3}
    at_true_rank = df[df["Rank"] == 3]
    assert (at_true_rank["RidgeComponentFMS"] > 0.99).all()
    np.testing.assert_allclose(
        at_true_rank["RidgeFitR2X"], at_true_rank["NoRidgeFitR2X"], atol=1e-3
    )


#: Canonical-fit provenance `_run_cpd_rank` stands in for the factors sidecar.
CANONICAL_PROVENANCE: dict[str, Any] = {
    "rank_selection": {"rise": {"rank": 5}, "cpd": {"rank": 2}},
    "complex_sep": "&",
    "communication_score": "expression_product",
    "hill_k": 0.5,
    "resulting_dims": {},
    "lr_pairs_annotated": 36,
    "r2x_pf2": 0.1,
}
CORE_COMPARISON = {"CoreFitR2X": 0.5, "CoreFMS": 1.0, "CoreRidgeFMS": 0.99}


def _run_cpd_rank(tensor: np.ndarray, output_path, **kwargs):
    """`run_cpd_rank_selection` on `tensor` in place of the canonical ALAD
    tensor, with the canonical fit's provenance and core comparison stubbed."""
    from dataclasses import replace
    from unittest.mock import patch

    import cccrise.runners.run_cpd_rank as runner
    from cccrise.runners.datasets import DATASETS

    with (
        patch.object(runner, "canonical_tensor", return_value=(tensor, None)),
        patch.object(runner, "read_metadata", return_value=CANONICAL_PROVENANCE),
        patch.object(runner, "core_comparison", return_value=CORE_COMPARISON),
        # DATASETS["alad"] points at the real (private, multi-GB) BAL
        # source file, which isn't present in CI; only the mocked tensor
        # above should matter here, so stub out the hash of that file.
        patch("cccrise.runners._cli.sha256_file", return_value="0" * 64),
    ):
        # CPU: the GPU extra (CuPy) isn't installed in CI.
        kwargs.setdefault("cp_device", "cpu")
        return runner.run_cpd_rank_selection(
            replace(DATASETS["alad"], backend=None), output_path=output_path, **kwargs
        )


def _run_start_sweep(tmp_path, true_rank: int = 3):
    """`_run_cpd_rank` over ranks 1-4 from three seeded starts (seeds 7-9),
    without imputation or the ridge comparison."""
    tensor = _simulate_cp_tensor((24, 10, 10, 36), rank=true_rank, random_state=22)
    return _run_cpd_rank(
        tensor,
        tmp_path / "cpd.parquet",
        ranks=[1, 2, 3, 4],
        n_starts=3,
        cp_n_iter_max=2000,
        cp_tol=1e-12,
        imputation_repeats=0,
        compare_ridge=False,
        random_state=7,
    )


def test_runner_fits_every_rank_from_seeded_starts(tmp_path):
    """One row per (rank, start), start 0 the production seed."""
    df, _out, sidecar = _run_start_sweep(tmp_path)
    assert df.groupby("Rank").size().to_dict() == {1: 3, 2: 3, 3: 3, 4: 3}
    assert set(df.loc[df["Start"] == 0, "Seed"]) == {7}
    assert df.groupby("Rank")["FitR2X"].median().is_monotonic_increasing
    assert df.loc[df["Rank"] == 1, "MaxCosA"].isna().all()
    assert df.loc[df["Rank"] > 1, [f"MaxCos{m}" for m in "ABCD"]].notna().all().all()
    text = sidecar.read_text()
    for field in ("n_starts: 3", "cp_dtype: float32", "cp_backend: numpy"):
        assert field in text
    assert "compress" not in text


def test_runner_writes_every_pair_of_starts_agreement(tmp_path):
    """The FMS of every pair of a rank's starts; at the true rank every start
    agrees."""
    from math import comb

    from cccrise.runners.datasets import DATASETS
    from cccrise.runners.provenance import sidecar_path

    _df, out, _sidecar = _run_start_sweep(tmp_path, true_rank=3)
    path = out.with_name(DATASETS["alad"].cpd_start_fms_filename())
    agreement = pd.read_parquet(path)
    assert len(agreement) == 4 * comb(3, 2)
    assert (agreement.loc[agreement["Rank"] == 3, "FMS"] > 0.99).all()
    assert sidecar_path(path).exists()


def test_runner_writes_per_split_imputation(tmp_path, monkeypatch):
    import cccrise.runners.run_cpd_rank as runner

    monkeypatch.setitem(runner.IMPUTATION_OPTIONS, "n_folds", 3)
    tensor = _simulate_cp_tensor((8, 6, 6, 12), rank=2, noise_std=0.3, random_state=3)
    _df, out, sidecar = _run_cpd_rank(
        tensor,
        tmp_path / "cpd.parquet",
        ranks=[1, 2, 3],
        cp_n_iter_max=200,
        cp_tol=1e-8,
        n_starts=1,
        imputation_repeats=2,
        compare_ridge=False,
    )
    from cccrise.runners.datasets import DATASETS
    from cccrise.runners.provenance import sidecar_path

    path = out.with_name(DATASETS["alad"].cpd_imputation_filename())
    imputation = pd.read_parquet(path)
    assert len(imputation) == 2 * 3 and imputation["ImputationR2X"].notna().all()
    assert imputation.groupby("Rank")["ImputationR2X"].mean().idxmax() == 2
    assert "imputation_repeats: 2" in sidecar.read_text()
    assert sidecar_path(path).exists()


def test_runner_ridge_comparison_adds_one_row_per_component(tmp_path):
    from cccrise.figures.common import per_rank
    from cccrise.runners.datasets import DATASETS

    tensor = _simulate_cp_tensor((8, 6, 6, 12), rank=2, noise_std=0.3, random_state=3)
    _df, out, sidecar = _run_cpd_rank(
        tensor,
        tmp_path / "cpd.parquet",
        ranks=[1, 2, 3],
        n_starts=1,
        cp_n_iter_max=200,
        cp_tol=1e-8,
        imputation_repeats=0,
    )
    ridge = pd.read_parquet(out.with_name(DATASETS["alad"].cpd_ridge_filename()))
    assert ridge.groupby("Rank").size().to_dict() == {1: 1, 2: 2, 3: 3}
    assert list(per_rank(ridge)["Rank"]) == [1, 2, 3]
    assert ridge["RidgeComponentFMS"].between(0.0, 1.0 + 1e-9).all()
    # The canonical core fit is compared at its own rank only.
    core_rows = ridge[list(CORE_COMPARISON)].notna().all(axis=1)
    assert set(ridge.loc[core_rows, "Rank"]) == {2}
    assert "compare_ridge: true" in sidecar.read_text()
