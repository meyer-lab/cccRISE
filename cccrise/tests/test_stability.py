"""Tests for cccrise.validation."""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import parafac
from tlviz.factor_tools import factor_match_score

from cccrise.ccc import calculate_r2x
from cccrise.utils import (
    _carry_paired_samples,
    _paired_conditions,
    resample_cells,
    resample_conditions,
    resample_tensor,
)
from cccrise.validation.component_stability import (
    CPD_SKIP_MODES,
    DEFAULT_AXES,
    RISE_SKIP_MODES,
    SEED_AXIS,
    STAGE_CPD,
    STAGE_RISE,
    _fit_cp,
    component_stability,
    fit_stability_pipeline,
    metric_interval,
    per_component_fms,
    reference_condition_rows,
    summarize_component_stability,
)

RISE_RANK = 3
CP_RANK = 2


def _random_factors(dims: tuple[int, ...], rank: int, random_state: int) -> list:
    rng = np.random.default_rng(random_state)
    return [rng.random((dim, rank)) for dim in dims]


def _simulate_adata(
    n_cond: int = 4,
    cells_per_cond: int = 12,
    random_state: int = 0,
) -> anndata.AnnData:
    """A tiny synthetic cohort the whole pipeline can run on.

    Two phenotype strata of equal size, so the stratified condition draw has
    something to stratify within, and three ligand-receptor pairs among six
    genes so the communication tensor's LR mode is longer than `CP_RANK`.
    """
    rng = np.random.default_rng(random_state)
    n_cells = n_cond * cells_per_cond
    counts = rng.poisson(lam=3.0, size=(n_cells, 6)).astype(np.float32)
    conditions = [f"S{i // cells_per_cond}" for i in range(n_cells)]
    obs = pd.DataFrame(
        {
            "sample": conditions,
            "group": [
                "ctrl" if int(c[1:]) < n_cond // 2 else "case" for c in conditions
            ],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=["L1", "R1", "L2", "R2", "L3", "R3"])
    adata = anndata.AnnData(X=csr_array(counts), obs=obs, var=var)
    adata.var["means"] = counts.mean(axis=0)
    return adata


def _lr_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": ["L1", "L2", "L3"],
            "receptor": ["R1", "R2", "R3"],
            "interaction_symbol": ["L1^R1", "L2^R2", "L3^R3"],
        }
    )


from typing import Any, cast


def _pipeline_kwargs(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "condition_column": "sample",
        "rise_rank": RISE_RANK,
        "cp_rank": CP_RANK,
        "group_column": "group",
        "rise_n_iter_max": 10,
        "cp_n_iter_max": 20,
        "init": "random",
        "svd": "randomized_svd",
    }
    return {**defaults, **overrides}


# =============================================================================
# Component matching
# =============================================================================


def test_per_component_fms_is_one_against_itself():
    factors = _random_factors((5, 4, 4, 7), rank=3, random_state=1)
    df = per_component_fms(factors, factors)

    assert list(df["Component"]) == [1, 2, 3]
    assert list(df["MatchedComponent"]) == [1, 2, 3]
    np.testing.assert_allclose(df["FMS"], 1.0, atol=1e-9)


@pytest.mark.parametrize("skip_modes", [(), (3,), (1, 2)])
def test_per_component_fms_mean_reproduces_tlviz_fms(skip_modes):
    """The per-component values are a decomposition of the aggregate FMS the
    Methods define, not a different metric, so their mean must equal it."""
    dims = (5, 4, 4, 7)
    a = _random_factors(dims, rank=3, random_state=2)
    b = _random_factors(dims, rank=3, random_state=3)

    kept = [m for m in range(len(dims)) if m not in skip_modes]
    reference = factor_match_score(
        CPTensor((np.ones(3), [a[m] for m in kept])),
        CPTensor((np.ones(3), [b[m] for m in kept])),
        consider_weights=False,
    )
    mine = per_component_fms(a, b, skip_modes)
    np.testing.assert_allclose(mine["FMS"].mean(), reference, atol=1e-9)


def test_per_component_fms_reports_the_runner_up_match():
    """A component's runner-up must be a different component."""
    a = _random_factors((6, 5, 5, 8), rank=3, random_state=4)
    b = _random_factors((6, 5, 5, 8), rank=3, random_state=5)
    df = per_component_fms(a, b)

    assert (df["RunnerUpComponent"] != df["MatchedComponent"]).all()
    assert (df["RunnerUpComponent"] >= 1).all()
    assert (df["RunnerUpComponent"] <= 3).all()
    assert df["RunnerUpFMS"].notna().all()
    assert (df["RunnerUpFMS"] >= 0.0).all()


def test_per_component_fms_compares_all_modes_by_default():
    dims = (5, 4, 3, 6)
    a = _random_factors(dims, rank=2, random_state=10)
    b = _random_factors(dims, rank=2, random_state=11)
    df = per_component_fms(a, b)
    for mode in range(len(dims)):
        assert df[f"SignedCosMode{mode}"].notna().all()


def test_per_component_fms_runner_up_is_absent_at_rank_one():
    a = _random_factors((6, 5, 5, 8), rank=1, random_state=6)
    df = per_component_fms(a, a)

    assert list(df["RunnerUpComponent"]) == [-1]
    assert df["RunnerUpFMS"].isna().all()


def test_per_component_fms_signed_cosines_are_nan_on_skipped_modes():
    """The congruence product is absolute, so a genuine sign reversal is only
    visible in the per-mode signed cosines."""
    a = _random_factors((6, 5, 5, 8), rank=2, random_state=7)
    b = [f.copy() for f in a]
    b[0] = -b[0]

    df = per_component_fms(a, b, skip_modes=(1, 2))

    np.testing.assert_allclose(df["FMS"], 1.0, atol=1e-9)
    np.testing.assert_allclose(df["SignedCosMode0"], -1.0, atol=1e-9)
    np.testing.assert_allclose(df["SignedCosMode3"], 1.0, atol=1e-9)
    assert df["SignedCosMode1"].isna().all()
    assert df["SignedCosMode2"].isna().all()


# =============================================================================
# Condition-mode row alignment
# =============================================================================


def test_reference_condition_rows_expands_repeat_draws_onto_their_source_rows():
    reference = np.array(["S0", "S1", "S2"])
    replicate = np.array(["S0", "S1", "S1__rep2", "S2"])

    np.testing.assert_array_equal(
        reference_condition_rows(reference, replicate), [0, 1, 1, 3 - 1]
    )


def test_reference_condition_rows_is_the_identity_on_an_unchanged_roster():
    labels = np.array(["S0", "S1", "S2"])
    np.testing.assert_array_equal(
        reference_condition_rows(labels, labels), np.arange(3)
    )


def test_reference_condition_rows_rejects_a_condition_absent_from_the_reference():
    with pytest.raises(ValueError, match="absent from the reference"):
        reference_condition_rows(np.array(["S0"]), np.array(["S0", "S9"]))


# =============================================================================
# Resamplers
# =============================================================================


def test_resample_cells_keeps_every_condition_and_recomputes_gene_means():
    """RISE centers each condition slice with `var["means"]` and the
    communication score adds it back, so a stale column biases both stages."""
    adata = _simulate_adata()
    resampled = resample_cells(adata, "sample", random_seed=0)

    assert resampled.n_obs == adata.n_obs
    assert set(resampled.obs["sample"]) == set(adata.obs["sample"])
    assert resampled.obs_names.is_unique
    assert isinstance(resampled.X, csr_array)
    np.testing.assert_allclose(
        resampled.var["means"].to_numpy(),
        np.asarray(resampled.X.mean(axis=0)).ravel(),
    )
    # With replacement, some cell is expected to be drawn more than once.
    assert len(set(resampled.obs_names)) == resampled.n_obs


def test_resample_cells_is_seed_reproducible():
    adata = _simulate_adata()
    a = resample_cells(adata, "sample", random_seed=1)
    b = resample_cells(adata, "sample", random_seed=1)
    c = resample_cells(adata, "sample", random_seed=2)

    assert (
        isinstance(a.X, csr_array)
        and isinstance(b.X, csr_array)
        and isinstance(c.X, csr_array)
    )
    np.testing.assert_allclose(a.X.toarray(), b.X.toarray())
    assert not np.allclose(a.X.toarray(), c.X.toarray())


def test_resample_conditions_preserves_the_count_and_relabels_repeat_draws():
    adata = _simulate_adata()
    n_conditions = adata.obs["sample"].nunique()

    labels = set()
    for seed in range(8):
        resampled = resample_conditions(adata, "sample", random_seed=seed)
        assert resampled.obs["sample"].nunique() == n_conditions
        labels |= set(resampled.obs["sample"])

    # A duplicated condition has to enter RISE as two separate slices, which
    # only happens if the repeat draw is relabeled.
    assert any("__rep" in label for label in labels)


def test_resample_conditions_recomputes_gene_means_over_the_resampled_cells():
    adata = _simulate_adata()
    resampled = resample_conditions(adata, "sample", random_seed=3)

    assert isinstance(resampled.X, csr_array)
    np.testing.assert_allclose(
        resampled.var["means"].to_numpy(),
        np.asarray(resampled.X.mean(axis=0)).ravel(),
    )


def test_resample_conditions_keeps_each_stratum_at_its_original_size():
    """An unstratified draw on a small cohort can lose a group entirely, which
    would register as instability caused by the draw rather than the cohort."""
    adata = _simulate_adata(n_cond=6)
    assert isinstance(adata.obs, pd.DataFrame)
    per_group = adata.obs.drop_duplicates("sample")["group"].value_counts()

    for seed in range(5):
        resampled = resample_conditions(
            adata, "sample", stratify_by="group", random_seed=seed
        )
        assert isinstance(resampled.obs, pd.DataFrame)
        drawn = resampled.obs.drop_duplicates("sample")["group"].value_counts()
        for group, count in per_group.items():
            assert drawn[group] == count


# =============================================================================
# Paired-patient resampling
# =============================================================================


def _simulate_adata_with_paired_patient(
    n_cond: int = 6, cells_per_cond: int = 8, random_state: int = 0
) -> anndata.AnnData:
    """`_simulate_adata`, plus a `patient` column pairing conditions S0 and S1
    to one patient; every other condition keeps its own distinct patient."""
    adata = _simulate_adata(
        n_cond=n_cond, cells_per_cond=cells_per_cond, random_state=random_state
    )
    patient_of = {f"S{i}": ("P_pair" if i < 2 else f"P{i}") for i in range(n_cond)}
    adata.obs["patient"] = adata.obs["sample"].map(patient_of)
    return adata


def test_paired_conditions_maps_only_conditions_sharing_a_multi_sample_patient():
    obs = pd.DataFrame(
        {"sample": ["S0", "S1", "S2", "S3"], "patient": ["P0", "P0", "P1", "P2"]}
    )
    labels = obs["sample"].to_numpy()
    pairs = _paired_conditions(obs, "sample", "patient", labels)

    assert pairs == {"S0": "S1", "S1": "S0"}


def test_carry_paired_samples_preserves_the_total_draw_count():
    """The compensating drop is what keeps the final number of patients in the
    resampling fixed: a carry-in must not grow the resample."""
    pair_of = {"S0": "S1", "S1": "S0"}
    drawn = np.array(["S0", "S2", "S3", "S2"])
    for seed in range(30):
        rng = np.random.default_rng(seed)
        result = _carry_paired_samples(drawn, pair_of, rng)
        assert len(result) == len(drawn)


def test_carry_paired_samples_leaves_exactly_one_of_the_pair():
    """Carrying S0's sibling in and immediately dropping one of the two
    (nothing else was drawn) must leave exactly one of the pair -- never
    neither, and never both."""
    pair_of = {"S0": "S1", "S1": "S0"}
    for seed in range(30):
        rng = np.random.default_rng(seed)
        result = _carry_paired_samples(["S0"], pair_of, rng)
        assert len(result) == 1
        assert result[0] in ("S0", "S1")


def test_resample_conditions_with_patient_by_keeps_the_condition_count_fixed():
    """Drawing one of a patient's two conditions carries the sibling in for
    free unless something else is dropped to compensate; the final resample
    must still hold exactly as many conditions as an unpaired draw would."""
    adata = _simulate_adata_with_paired_patient()
    n_conditions = adata.obs["sample"].nunique()

    for seed in range(20):
        resampled = resample_conditions(
            adata, "sample", random_seed=seed, patient_by="patient"
        )
        assert resampled.obs["sample"].nunique() == n_conditions


def test_resample_conditions_with_patient_by_can_carry_the_sibling_in():
    """Demonstrates the fix itself: a replicate can hold both of a patient's
    samples, which an unpaired per-condition bootstrap draws independently."""
    adata = _simulate_adata_with_paired_patient()

    carried = False
    for seed in range(50):
        resampled = resample_conditions(
            adata, "sample", random_seed=seed, patient_by="patient"
        )
        drawn = {label.split("__rep")[0] for label in resampled.obs["sample"]}
        if {"S0", "S1"} <= drawn:
            carried = True
            break
    assert carried


def test_resample_conditions_without_patient_by_is_unaffected():
    """`patient_by=None` (the default) must reproduce the unpaired behavior
    exactly, seed for seed."""
    adata = _simulate_adata_with_paired_patient()
    a = resample_conditions(adata, "sample", random_seed=5)
    b = resample_conditions(adata, "sample", random_seed=5, patient_by=None)

    a_obs = cast(pd.DataFrame, a.obs)
    b_obs = cast(pd.DataFrame, b.obs)
    pd.testing.assert_frame_equal(
        a_obs.reset_index(drop=True), b_obs.reset_index(drop=True)
    )


def test_carry_paired_samples_drops_only_from_the_siblings_stratum():
    """A carried-in sibling grows its own stratum, so the compensating drop
    must come from that stratum or the strata drift off their sizes."""
    pair_of = {"S0": "S3", "S3": "S0"}
    stratum_of = {"S0": "ctrl", "S1": "ctrl", "S2": "ctrl", "S3": "case"}
    drawn = ["S0", "S1", "S2", "S3"]
    for seed in range(30):
        rng = np.random.default_rng(seed)
        result = _carry_paired_samples(drawn, pair_of, rng, stratum_of)
        assert sorted(stratum_of[c] for c in result) == sorted(
            stratum_of[c] for c in drawn
        )


def test_resample_conditions_with_patient_by_keeps_each_stratum_size():
    """Pairing S0 (ctrl) with S3 (case) puts the pair across strata, the case
    where a drop from the whole draw would unbalance the groups."""
    adata = _simulate_adata(n_cond=6)
    adata.obs["patient"] = adata.obs["sample"].replace({"S3": "S0"})
    assert isinstance(adata.obs, pd.DataFrame)
    per_group = adata.obs.drop_duplicates("sample")["group"].value_counts()

    for seed in range(30):
        resampled = resample_conditions(
            adata, "sample", stratify_by="group", random_seed=seed, patient_by="patient"
        )
        assert isinstance(resampled.obs, pd.DataFrame)
        drawn = resampled.obs.drop_duplicates("sample")["group"].value_counts()
        for group, count in per_group.items():
            assert drawn[group] == count


# =============================================================================
# The replicate loop
# =============================================================================


def test_component_stability_emits_one_row_per_axis_stage_and_component():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    df = component_stability(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=("cells", "conditions", "lr_pairs", SEED_AXIS),
        n_replicates=2,
        random_state=0,
    )

    counts = df.groupby(["Axis", "Stage"]).size()
    assert counts[("cells", STAGE_RISE)] == 2 * RISE_RANK
    assert counts[("cells", STAGE_CPD)] == 2 * CP_RANK
    assert counts[("conditions", STAGE_RISE)] == 2 * RISE_RANK
    assert counts[("conditions", STAGE_CPD)] == 2 * CP_RANK
    assert counts[("lr_pairs", STAGE_CPD)] == 2 * CP_RANK
    # The seed axis refits the whole pipeline, so unlike the permutation floor
    # it replaced it has something to say about RISE components too.
    assert counts[(SEED_AXIS, STAGE_RISE)] == 2 * RISE_RANK
    assert counts[(SEED_AXIS, STAGE_CPD)] == 2 * CP_RANK

    assert df["FMS"].between(0.0, 1.0 + 1e-9).all()


def test_component_stability_gives_the_lr_pairs_axis_no_rise_rows():
    """Bootstrapping the tensor's ligand-receptor mode leaves RISE untouched,
    so there is nothing to say about RISE components on that axis."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    df = component_stability(
        adata, reference, lr_pairs, **kwargs, axes=("lr_pairs",), n_replicates=1
    )

    assert set(df["Stage"]) == {STAGE_CPD}


def test_component_stability_shares_one_seed_between_a_replicates_two_stages():
    """The point of the shared loop: a replicate's RISE and CPD rows have to
    describe the same resampled dataset, not two independent draws."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    df = component_stability(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=("cells",),
        n_replicates=2,
        random_state=7,
    )

    per_replicate = df.groupby("Replicate")["Seed"].nunique()
    assert (per_replicate == 1).all()
    assert sorted(df["Seed"].unique()) == [7 * 1000 + 1, 7 * 1000 + 2]


def test_component_stability_is_reproducible_from_its_random_state():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    run_kwargs = {"axes": ("cells",), "n_replicates": 1, "random_state": 0}
    a = component_stability(adata, reference, lr_pairs, **kwargs, **run_kwargs)
    b = component_stability(adata, reference, lr_pairs, **kwargs, **run_kwargs)

    pd.testing.assert_frame_equal(a, b)


def test_component_stability_carries_the_per_replicate_fit_diagnostics():
    """A wide FMS spread traceable to a poor or degenerate fit has to be
    distinguishable from genuine component instability."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    df = component_stability(
        adata, reference, lr_pairs, **kwargs, axes=("cells",), n_replicates=1
    )

    cpd = df[df["Stage"] == STAGE_CPD]
    assert cpd["R2X"].notna().all()
    assert cpd["CoreConsistency"].notna().all()
    assert cpd["MaxModesCollinear"].notna().all()
    # The diagnostics are replicate-level scalars repeated on each row.
    assert cpd["CoreConsistency"].nunique() == 1
    # RISE rows carry their own R2X but no CP-specific diagnostics.
    rise = df[df["Stage"] == STAGE_RISE]
    assert rise["R2X"].notna().all()
    assert rise["CoreConsistency"].isna().all()


# =============================================================================
# GPU backend and CANDELINC compression (ported from PR #182)
# =============================================================================


def test_fit_cp_diagnostics_match_between_core_and_decompressed_space():
    """`_fit_cp` computes R2X and diagnostics cheaply in the CANDELINC core,
    before decompressing the factors it returns. Those values must equal what
    recomputing them against the decompressed factors and the full tensor
    would give -- the invariance `rank_selection.candelinc_compress`
    documents, and the reason it is safe to read them off the core at all."""
    rng = np.random.default_rng(21)
    tensor = np.abs(rng.random((20, 9, 9, 30)))

    weights, factors, r2x, diagnostics = _fit_cp(
        tensor,
        rank=3,
        n_iter_max=200,
        tol=1e-8,
        init="svd",
        svd="randomized_svd",
        random_state=0,
        compress=True,
        oversample=5,
    )

    for f, dim in zip(factors, tensor.shape, strict=True):
        assert f.shape == (dim, 3)

    assert r2x == pytest.approx(calculate_r2x(weights, factors, tensor), abs=1e-6)

    from cccrise.rank_selection import cp_core_consistency, cpd_component_collinearity

    assert diagnostics["CoreConsistency"] == pytest.approx(
        cp_core_consistency(tensor, weights, factors), abs=1e-6
    )
    recomputed = cpd_component_collinearity({3: (weights, factors)}).iloc[0]
    assert diagnostics["MaxModesCollinear"] == int(recomputed["MaxModesCollinear"])


def test_fit_stability_pipeline_runs_with_compression_disabled():
    """`compress=False` must still be a supported code path, not just the
    default's untested complement."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()

    fit = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, compress=False
    )
    assert 0.0 <= fit["r2x_cp"] <= 1.0 + 1e-9


def test_fit_stability_pipeline_accepts_an_explicit_cpu_backend():
    """`backend` must reach `rise_store_r2x` without erroring on a plain CPU
    box, which is what every CI run and every cohort-scale fit uses."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()

    fit = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, backend="cpu"
    )
    assert 0.0 <= fit["r2x_rise"] <= 1.0 + 1e-9


def test_component_stability_runs_with_compression_disabled():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(),
        lr_pairs,
        **kwargs,
        random_state=0,
        keep_tensor=True,
        compress=False,
    )

    df = component_stability(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=("cells", "lr_pairs"),
        n_replicates=1,
        compress=False,
    )
    assert df["FMS"].between(0.0, 1.0 + 1e-9).all()


# =============================================================================
# The seed-refit ceiling
# =============================================================================


def test_seed_axis_is_run_by_default():
    """No other axis's FMS is interpretable without the ceiling beside it, and
    an absent ceiling shows up only as a column of NaNs -- easy to miss."""
    assert SEED_AXIS in DEFAULT_AXES


def test_seed_axis_emits_both_stages_with_real_cohort_sizes():
    """The axis it replaced decomposed the reference tensor and so could only
    describe CPD components, with no cell or condition count to report."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    df = component_stability(
        adata, reference, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=2
    )

    assert set(df["Stage"]) == {STAGE_RISE, STAGE_CPD}
    assert (df["NCells"] == adata.n_obs).all()
    assert (df["NConditions"] == adata.obs["sample"].nunique()).all()


def test_seed_axis_reproduces_the_reference_when_it_draws_the_reference_seed():
    """Proves the axis varies the seed and nothing else.

    `replicate_seed(0, 0) == 1`, so a reference fit at `random_state=1` is the
    fit the first replicate re-derives. If the axis were perturbing the data
    as well, the two would not land on top of each other.
    """
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=1, keep_tensor=True
    )

    df = component_stability(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=(SEED_AXIS,),
        n_replicates=1,
        random_state=0,
    )

    assert (df["FMS"] > 0.99).all()


def test_seed_axis_leaves_the_callers_adata_untouched():
    """`import_data.add_cond_idxs` writes into `obs` in place and
    `parafac2.store_pf2` writes `uns`/`varm`, so the seed axis's identity
    resampler has to copy. Without the copy the caller's object would carry
    the first replicate's fit and every later axis would resample from it."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    component_stability(
        adata, reference, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
    )

    assert "Pf2_A" not in adata.uns
    assert "condition_unique_idxs" not in adata.obs.columns


def test_seed_axis_does_not_require_a_retained_tensor():
    """It refits the pipeline rather than decomposing the reference tensor,
    so the guard that gates 'lr_pairs' must not gate it."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=False
    )

    df = component_stability(
        adata, reference, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
    )

    assert not df.empty


def test_component_stability_rejects_an_unknown_axis():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    with pytest.raises(ValueError, match="Unknown stability axes"):
        component_stability(
            adata, reference, lr_pairs, **kwargs, axes=("genes",), n_replicates=1
        )

    # The retired permutation floor is an unknown axis now, not a silently
    # accepted one: a stale command line should fail rather than produce a
    # table whose baseline column means the opposite of what it used to.
    with pytest.raises(ValueError, match="Unknown stability axes"):
        component_stability(
            adata, reference, lr_pairs, **kwargs, axes=("permuted",), n_replicates=1
        )


def test_component_stability_requires_a_retained_tensor_for_tensor_only_axes():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=False
    )

    with pytest.raises(ValueError, match="keep_tensor=True"):
        component_stability(
            adata, reference, lr_pairs, **kwargs, axes=("lr_pairs",), n_replicates=1
        )


def test_component_stability_rejects_a_replicate_with_a_different_lr_roster():
    """A silent roster difference is the dangerous case: matching would line
    up different interactions row by row and return a plausible number."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )
    shortened = {**reference, "lr_symbols": reference["lr_symbols"][:-1]}

    with pytest.raises(ValueError, match="different ligand-receptor"):
        component_stability(
            adata, shortened, lr_pairs, **kwargs, axes=("cells",), n_replicates=1
        )


def test_component_stability_rejects_a_replicate_with_a_different_gene_roster():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )
    shortened = {**reference, "gene_names": reference["gene_names"][:-1]}

    with pytest.raises(ValueError, match="different genes"):
        component_stability(
            adata, shortened, lr_pairs, **kwargs, axes=("cells",), n_replicates=1
        )


def test_component_stability_holds_out_expected_modes():
    """RISE holds out Mode 1 (B basis); CPD holds out Modes 1 and 2 (D, E)."""
    assert RISE_SKIP_MODES == (1,)
    assert CPD_SKIP_MODES == (1, 2)

    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, keep_tensor=True
    )

    df = component_stability(
        adata, reference, lr_pairs, **kwargs, axes=("cells",), n_replicates=1
    )

    rise = df[df["Stage"] == STAGE_RISE]
    assert rise["SignedCosMode0"].notna().all()
    assert rise["SignedCosMode1"].isna().all()
    assert rise["SignedCosMode2"].notna().all()

    cpd = df[df["Stage"] == STAGE_CPD]
    assert cpd["SignedCosMode0"].notna().all()
    assert cpd["SignedCosMode1"].isna().all()
    assert cpd["SignedCosMode2"].isna().all()
    assert cpd["SignedCosMode3"].notna().all()


# =============================================================================
# Summaries & Decompositions
# =============================================================================


def test_parafac_supports_randomized_svd():
    rng = np.random.default_rng(4)
    tensor = np.abs(rng.random((6, 5, 5, 20)))
    w, f = parafac(
        tensor,
        rank=2,
        n_iter_max=50,
        tol=1e-6,
        init="svd",
        svd="randomized_svd",
        random_state=42,
    )
    r2x = calculate_r2x(w, f, tensor)
    assert w.shape == (2,)
    assert len(f) == 4
    assert 0.0 <= r2x <= 1.0


def test_resample_tensor():
    rng = np.random.default_rng(0)
    tensor = rng.random((3, 4, 4, 10))
    resampled, indices = resample_tensor(tensor, random_seed=42, return_indices=True)
    assert resampled.shape == tensor.shape
    assert len(indices) == 10
    np.testing.assert_allclose(resampled, tensor[..., indices])


def test_metric_interval_averages_replicate_rows_before_taking_the_band():
    """A replicate contributes one value per group, so within-replicate
    variation never leaks into the reported spread."""
    rows = [
        {"Resampling": scheme, "Replicate": b, "Rank": rank, "Fold": fold, "M": value}
        for scheme, spread in [("cells", 0.01), ("conditions", 0.10)]
        for b, offset in enumerate([-1.0, 0.0, 1.0])
        for rank in (5, 10)
        for fold, value in enumerate([rank + spread * offset] * 2)
    ]
    summary = metric_interval(
        pd.DataFrame(rows), "M", by=("Resampling", "Rank")
    ).set_index(["Resampling", "Rank"])

    assert summary.loc[("cells", 5), "NReplicates"] == 3
    assert summary.loc[("cells", 5), "Median"] == 5.0
    cells_band = summary.loc[("cells", 5), "Max"] - summary.loc[("cells", 5), "Min"]
    cond_band = (
        summary.loc[("conditions", 5), "Max"] - summary.loc[("conditions", 5), "Min"]
    )
    assert cond_band > cells_band
    assert cells_band < 5.0


def test_metric_interval_groups_by_the_keys_it_is_given():
    """Grouped per component the fold-mean is an identity; grouped per stage it
    averages components within a replicate, which is the aggregate FMS."""
    rows = [
        {
            "Axis": "cells",
            "Stage": STAGE_CPD,
            "Replicate": b,
            "Component": component,
            "FMS": 0.5 + 0.1 * component,
        }
        for b in range(3)
        for component in (1, 2)
    ]
    df = pd.DataFrame(rows)

    per_component = metric_interval(df, "FMS")
    assert len(per_component) == 2
    np.testing.assert_allclose(sorted(per_component["Mean"]), [0.6, 0.7])

    aggregate = metric_interval(df, "FMS", by=("Axis", "Stage"))
    assert len(aggregate) == 1
    np.testing.assert_allclose(aggregate["Mean"].iloc[0], 0.65)


def _stability_frame(with_ceiling: bool) -> pd.DataFrame:
    """A summary-ready table whose two components need different ceilings.

    Component 1 reseeds almost perfectly but the cell bootstrap moves it well
    below that, so the perturbation genuinely cost it something. Component 2
    reseeds worse than component 1 scores under resampling -- it is fuzzy at
    this rank however it is fit -- so its lower observed FMS is not the data's
    doing. A ceiling pooled across components would put the band edge at 0.45
    and call both components fine; matching per component separates them, so
    these values are what makes that behaviour testable rather than incidental.
    """
    rows = [
        {
            "Axis": "cells",
            "Replicate": b,
            "Stage": STAGE_CPD,
            "Component": component,
            "MatchedComponent": component,
            "FMS": fms,
            "RunnerUpComponent": 3 - component,
            "RunnerUpFMS": 0.1,
        }
        for component, values in [(1, [0.9, 0.92, 0.95]), (2, [0.5, 0.55, 0.6])]
        for b, fms in enumerate(values)
    ]
    if with_ceiling:
        rows += [
            {
                "Axis": SEED_AXIS,
                "Replicate": b,
                "Stage": STAGE_CPD,
                "Component": component,
                "MatchedComponent": component,
                "FMS": fms,
                "RunnerUpComponent": 3 - component,
                "RunnerUpFMS": 0.1,
            }
            for component, fms in [(1, 0.99), (2, 0.45)]
            for b in range(3)
        ]
    return pd.DataFrame(rows)


def test_summarize_component_stability_scores_each_component_against_the_seed_band():
    summary = summarize_component_stability(_stability_frame(with_ceiling=True))
    summary = summary.set_index("Component")

    assert set(summary.index) == {1, 2}
    np.testing.assert_allclose(summary.loc[1, "SeedP5FMS"], 0.99)
    np.testing.assert_allclose(summary.loc[2, "SeedP5FMS"], 0.45)
    # Component 1 reseeds better than any bootstrap replicate reproduced it,
    # so the draw moved it; component 2 never drops below what reseeding alone
    # costs it, which the aggregate FMS alone would read as the worse result.
    assert summary.loc[1, "FracWithinSeedBand"] == 0.0
    assert summary.loc[2, "FracWithinSeedBand"] == 1.0
    np.testing.assert_allclose(summary.loc[1, "MedianFMS"], 0.92)
    np.testing.assert_allclose(summary.loc[1, "MedianRunnerUpFMS"], 0.1)


def test_summarize_component_stability_matches_the_seed_ceiling_per_component():
    """Pooling the ceiling across components would hide the distinction.

    A single stage-wide band edge here is 0.45, which every observed replicate
    of both components clears -- so both would read 1.0 and component 1's real
    sensitivity to the draw would vanish.
    """
    df = _stability_frame(with_ceiling=True)
    pooled_edge = np.percentile(df[df["Axis"] == SEED_AXIS]["FMS"], 5)
    assert (df[df["Axis"] == "cells"]["FMS"] >= pooled_edge).all()

    summary = summarize_component_stability(df).set_index("Component")
    assert summary.loc[1, "FracWithinSeedBand"] == 0.0


def test_summarize_component_stability_reports_nan_when_no_seed_axis_was_run():
    """0.0 would report every component as having failed a test that was
    never performed."""
    summary = summarize_component_stability(_stability_frame(with_ceiling=False))

    assert summary["SeedMedianFMS"].isna().all()
    assert summary["SeedP5FMS"].isna().all()
    assert summary["FracWithinSeedBand"].isna().all()


def test_summarize_component_stability_survives_a_csv_round_trip(tmp_path):
    """The baseline axis is named 'seed_only', not 'null', because `read_csv`
    parses the latter as NaN and silently erases the baseline.

    Asserted at the cause as well as the consequence: a future rename that
    happens to land on one of pandas' NA strings should fail here rather than
    downstream, as an unexplained column of NaNs.
    """
    from pandas._libs.parsers import STR_NA_VALUES

    assert SEED_AXIS not in STR_NA_VALUES

    path = tmp_path / "stability.csv"
    _stability_frame(with_ceiling=True).to_csv(path, index=False)

    summary = summarize_component_stability(pd.read_csv(path))

    assert summary["SeedP5FMS"].notna().all()
    assert summary["FracWithinSeedBand"].notna().all()


def test_reference_condition_rows_matches_on_labels_not_positions():
    """The map is a label lookup, so a roster whose sorted order permutes the
    reference's still lands on the right rows.

    `P10` sorts before `P2` as a string but after it as a number, so a roster
    that reorders the two is exactly the case a positional map would get
    wrong -- and it would get it wrong quietly, by comparing one patient's
    factor row against another's.
    """
    reference = np.array(["P1", "P2", "P10"])
    replicate = np.array(["P10", "P1", "P1__rep2", "P2"])

    np.testing.assert_array_equal(
        reference_condition_rows(reference, replicate), [2, 0, 0, 1]
    )


def test_fit_stability_pipeline_labels_conditions_in_row_order():
    """`condition_labels` must name mode-0 row for row.

    `add_cond_idxs` orders the rows by a sort in the condition column's own
    dtype, so an integer column puts row 2 at patient ``10``. Deriving the
    labels with an independent string sort would instead call that row ``2``,
    and `reference_condition_rows` -- which reads this array as positionally
    aligned to the rows -- would then match replicate patients onto the wrong
    reference rows without raising.
    """
    adata = _simulate_adata()
    # Sample IDs whose numeric and string sorts disagree: [1, 2, 10, 20]
    # against ['1', '10', '2', '20'].
    numeric_ids = {"S0": 1, "S1": 2, "S2": 10, "S3": 20}
    adata.obs["patient"] = adata.obs["sample"].map(numeric_ids).astype(int)
    kwargs = _pipeline_kwargs(condition_column="patient")

    fit = fit_stability_pipeline(adata, _lr_pairs(), **kwargs, random_state=0)

    # `add_cond_idxs` wrote the row each cell's condition owns; the label of
    # row i is the patient of any cell carrying index i.
    rows = adata.obs["condition_unique_idxs"].to_numpy()
    patients = adata.obs["patient"].to_numpy().astype(str)
    for row, patient in zip(rows, patients, strict=True):
        assert fit["condition_labels"][row] == patient
    assert len(fit["condition_labels"]) == fit["rise_factors"][0].shape[0]


def test_fit_stability_pipeline_gives_a_twice_drawn_condition_two_equal_rows():
    """A repeat draw is fit as its own condition off identical cells, so its
    two mode-0 rows coincide and both expand from the one reference row."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)
    replicate = fit_stability_pipeline(
        resample_conditions(adata, "sample", stratify_by="group", random_seed=2),
        lr_pairs,
        **kwargs,
        random_state=0,
    )

    labels = list(replicate["condition_labels"])
    assert labels == ["S0", "S0__rep2", "S2", "S3"]

    A = replicate["rise_factors"][0]
    np.testing.assert_allclose(A[labels.index("S0")], A[labels.index("S0__rep2")])

    rows = reference_condition_rows(
        reference["condition_labels"], replicate["condition_labels"]
    )
    np.testing.assert_array_equal(rows, [0, 0, 2, 3])
