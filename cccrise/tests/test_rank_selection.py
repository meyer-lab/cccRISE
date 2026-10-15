"""Tests for cccrise.rank_selection.

Only the CPD stage is covered here. The RISE-side bi-cross-validation and its
compression moved to `scrise`, and are tested there; what remains is the
interaction tensor's own rank criteria and the CANDELINC compression they run
on, which is a 4-mode CP problem with no upstream home.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from cccrise.ccc_rise import CP_FIT_OPTIONS, RISE_FIT_OPTIONS
from cccrise.rank_selection import (
    calculate_r2x,
    candelinc_compress,
    candelinc_r2x,
    cp_core_consistency,
    cp_core_consistency_sweep,
    cpd_component_collinearity,
    matched_congruence,
    screen_pairs,
    select_rank_by_rule,
)


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
    assert df.loc[df["Rank"] == 3, "MinCosineProduct"].item() == pytest.approx(-1)


def test_cpd_component_collinearity_one_mode_sign_is_not_canceling():
    """A pair opposite in one mode only is a shared (sign-flipped) factor,
    not a cancellation: the sign can be moved to any other mode."""
    rng = np.random.default_rng(1)
    factors = [rng.normal(size=(d, 3)) for d in (5, 6, 6, 8)]
    factors[1][:, 1] = -factors[1][:, 0]
    df = cpd_component_collinearity({3: (np.ones(3), factors)})
    assert df["MaxModesCollinear"].item() == 1
    assert df["WorstPairCanceling"].item() is False


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
    """`compress=False` and the compressed path the runners take must agree.

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
        with (
            patch.object(
                runner, "build_interaction_tensor", return_value=(tensor, facts)
            ),
            # DATASETS["alad"] points at the real (private, multi-GB) BAL
            # source file, which isn't present in CI; only the mocked tensor
            # above should matter here, so stub out the hash of that file.
            patch("cccrise.runners._cli.sha256_file", return_value="0" * 64),
        ):
            df, out_path, sidecar = runner.run_cpd_rank_selection(
                DATASETS["alad"],
                rise_rank=5,
                ranks=[1, 2, 3, 4, 5],
                cp_n_iter_max=2000,
                cp_tol=1e-12,
                compress=compress,
                output_path=tmp_path / f"cpd_{compress}.parquet",
            )
        assert out_path.exists() and sidecar.exists()
        # The RISE fit options are hard-coded, so the sidecar must say so.
        recorded = yaml.safe_load(sidecar.read_text())
        assert recorded | RISE_FIT_OPTIONS | {"compress": compress} == recorded
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


def _pair_factors(cosines, rng, rank=3, dims=(12, 10, 10, 30)):
    """Factors whose components 0 and 1 have the given per-mode cosines and
    whose third component is generic."""
    factors = []
    for dim, cos in zip(dims, cosines, strict=True):
        f = np.linalg.qr(rng.normal(size=(dim, rank)))[0]
        f[:, 1] = cos * f[:, 0] + np.sqrt(1 - cos**2) * f[:, 1]
        factors.append(f)
    return factors


def test_screen_pairs_reads_each_rule_from_the_cosines():
    rng = np.random.default_rng(0)
    assert screen_pairs(_pair_factors((0.95, 0.2, 0.1, 0.0), rng)) == (False, False, [])
    assert screen_pairs(_pair_factors((0.95, 0.95, 0.1, 0.0), rng))[1]
    assert screen_pairs(_pair_factors((0.85, 0.85, 0.1, 0.0), rng))[2] == [(0, 1)]
    canceling = _pair_factors((-0.95, -0.95, -0.95, 0.95), rng)
    assert screen_pairs(canceling)[0]
    # one mode's sign alone is a convention, not a cancellation
    assert not screen_pairs(_pair_factors((-0.95, 0.1, 0.1, 0.1), rng))[0]


def test_matched_congruence_ignores_order_and_sign():
    rng = np.random.default_rng(1)
    factors = [rng.normal(size=(dim, 3)) for dim in (8, 6, 6, 20)]
    shuffled = [f[:, [2, 0, 1]] * np.array([1, -1, 1]) for f in factors]
    shuffled[1][:, 1] *= -1  # the sign moves to another mode
    np.testing.assert_allclose(matched_congruence(factors, shuffled), 1.0)


def _starts(fits, tensor):
    """Starts from (weights, factors) fits; a bare factor list gets unit weights."""
    fits = [
        fit if isinstance(fit, tuple) else (np.ones(fit[0].shape[1]), fit)
        for fit in fits
    ]
    return [(w, f, calculate_r2x(w, f, tensor)) for w, f in fits]


def test_select_rank_by_rule_keeps_the_largest_passing_rank():
    rng = np.random.default_rng(2)
    tensor = _simulate_cp_tensor((12, 10, 10, 30), rank=2, noise_std=0.01)
    good = [tuple(_cp_fit(tensor, r)[:2]) for r in (1, 2)]
    # rank 3: every start splits one program over two components
    bad = _pair_factors((0.99, 0.99, 0.3, 0.2), rng)
    by_rank = [(1, _starts([good[0]] * 10, tensor)), (2, _starts([good[1]] * 10, tensor)),
               (3, _starts([bad] * 10, tensor))]  # fmt: skip
    rank, table, (_weights, factors) = select_rank_by_rule(tensor, by_rank)
    assert rank == 2
    assert table["Passes"].tolist() == [True, True, False]
    np.testing.assert_allclose(matched_congruence(good[1][1], factors), 1.0)
    early_rank, early, _fit = select_rank_by_rule(tensor, by_rank, stop_at_failure=True)
    assert early_rank == 2 and len(early) == 3


def test_select_rank_by_rule_needs_nine_of_ten_starts():
    rng = np.random.default_rng(3)
    tensor = _simulate_cp_tensor((12, 10, 10, 30), rank=2, noise_std=0.01)
    good = tuple(_cp_fit(tensor, 2)[:2])
    bad = _pair_factors((0.99, 0.99, 0.3, 0.2), rng, rank=2)
    one_bad = _starts([good] * 9 + [bad], tensor)
    two_bad = _starts([good] * 8 + [bad] * 2, tensor)
    assert select_rank_by_rule(tensor, [(2, one_bad)])[1]["Passes"].item()
    assert not select_rank_by_rule(tensor, [(2, two_bad)])[1]["Passes"].item()


def test_candelinc_energy_keeps_that_share_of_every_mode():
    tensor = _simulate_cp_tensor((30, 12, 12, 40), rank=3, noise_std=0.3)
    _core, bases, lost_sq = candelinc_compress(tensor, max_rank=3, energy=0.999)
    _small, small_bases, small_lost = candelinc_compress(tensor, max_rank=3)
    assert all(
        b.shape[1] >= s.shape[1] for b, s in zip(bases, small_bases, strict=True)
    )
    assert lost_sq / np.sum(tensor**2) <= 4e-3 < small_lost / np.sum(tensor**2)
