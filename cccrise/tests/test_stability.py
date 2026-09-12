"""Tests for cccrise.stability."""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import parafac
from tlviz.factor_tools import factor_match_score

from cccrise.ccc import calculate_r2x
from cccrise.stability.component_stability import (
    CPD_SKIP_MODES,
    PERMUTED_AXIS,
    RISE_SKIP_MODES,
    STAGE_CPD,
    STAGE_RISE,
    component_stability,
    fit_stability_pipeline,
    metric_interval,
    per_component_fms,
    reference_condition_rows,
    summarize_component_stability,
)
from cccrise.utils import resample_cells, resample_conditions, resample_tensor

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


from typing import Any


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
        axes=("cells", "conditions", "lr_pairs", PERMUTED_AXIS),
        n_replicates=2,
        random_state=0,
    )

    counts = df.groupby(["Axis", "Stage"]).size()
    assert counts[("cells", STAGE_RISE)] == 2 * RISE_RANK
    assert counts[("cells", STAGE_CPD)] == 2 * CP_RANK
    assert counts[("conditions", STAGE_RISE)] == 2 * RISE_RANK
    assert counts[("conditions", STAGE_CPD)] == 2 * CP_RANK
    assert counts[("lr_pairs", STAGE_CPD)] == 2 * CP_RANK
    assert counts[(PERMUTED_AXIS, STAGE_CPD)] == 2 * CP_RANK

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


def _stability_frame(with_floor: bool) -> pd.DataFrame:
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
        # Component 1 is reproducible, component 2 hovers near chance.
        for component, values in [(1, [0.9, 0.92, 0.95]), (2, [0.2, 0.25, 0.3])]
        for b, fms in enumerate(values)
    ]
    if with_floor:
        rows += [
            {
                "Axis": PERMUTED_AXIS,
                "Replicate": b,
                "Stage": STAGE_CPD,
                "Component": component,
                "MatchedComponent": component,
                "FMS": 0.3,
                "RunnerUpComponent": 3 - component,
                "RunnerUpFMS": 0.1,
            }
            for component in (1, 2)
            for b in range(3)
        ]
    return pd.DataFrame(rows)


def test_summarize_component_stability_scores_each_component_against_the_floor():
    summary = summarize_component_stability(_stability_frame(with_floor=True))
    summary = summary.set_index("Component")

    assert set(summary.index) == {1, 2}
    np.testing.assert_allclose(summary.loc[1, "NullP95FMS"], 0.3)
    # The reproducible component clears the floor every time; the other never
    # does, which the aggregate FMS alone would not reveal.
    assert summary.loc[1, "FracAboveNull"] == 1.0
    assert summary.loc[2, "FracAboveNull"] == 0.0
    np.testing.assert_allclose(summary.loc[1, "MedianFMS"], 0.92)
    np.testing.assert_allclose(summary.loc[1, "MedianRunnerUpFMS"], 0.1)


def test_summarize_component_stability_reports_nan_when_no_floor_was_run():
    """0.0 would report every component as having failed a test that was
    never performed."""
    summary = summarize_component_stability(_stability_frame(with_floor=False))

    assert summary["NullMedianFMS"].isna().all()
    assert summary["NullP95FMS"].isna().all()
    assert summary["FracAboveNull"].isna().all()


def test_summarize_component_stability_survives_a_csv_round_trip(tmp_path):
    """The floor axis is named 'permuted', not 'null', because `read_csv`
    parses the latter as NaN and silently erases the floor."""
    path = tmp_path / "stability.csv"
    _stability_frame(with_floor=True).to_csv(path, index=False)

    summary = summarize_component_stability(pd.read_csv(path))

    assert summary["NullP95FMS"].notna().all()
    assert summary["FracAboveNull"].notna().all()
