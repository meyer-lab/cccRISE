"""Tests for cccrise.validation."""

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import parafac
from tlviz.factor_tools import factor_match_score

from cccrise.ccc_rise import fit_cp, run_ccc_rise_workflow
from cccrise.factors import save_ccc_rise_fit
from cccrise.rank_selection import calculate_r2x
from cccrise.runners._cli import CP_TOL, RISE_TOL
from cccrise.utils import (
    jackknife_units,
    leave_out_patient,
    resample_tensor,
)
from cccrise.validation.stability import (
    ANOVA_COLUMNS,
    CPD_SKIP_MODES,
    DEFAULT_AXES,
    JACKKNIFE_AXIS,
    RISE_SKIP_MODES,
    SEED_AXIS,
    STAGE_CPD,
    STAGE_RISE,
    _attach_anova,
    _eta_squared_interval,
    axis_skip_modes,
    fit_stability_pipeline,
    metric_interval,
    per_component_fms,
    relabel_workflow_fit,
    stability_replicates,
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
        "rise_tol": RISE_TOL,
        "cp_tol": CP_TOL,
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


def test_relabel_workflow_fit_restores_the_core_numbering(tmp_path):
    fit = run_ccc_rise_workflow(
        _simulate_adata(),
        RISE_RANK,
        _lr_pairs(),
        cp_rank=CP_RANK,
        condition_column="sample",
        random_state=0,
        keep_tensor=True,
        backend="cpu",
        rise_n_iter_max=10,
        cp_n_iter_max=20,
        rise_tol=RISE_TOL,
        cp_tol=CP_TOL,
    )
    core = tmp_path / "core.nc"
    save_ccc_rise_fit(core, fit["adata"], fit, "sample")
    adata = fit["adata"]
    expected = {
        "Pf2_A": adata.uns["Pf2_A"].copy(),
        "projections": np.asarray(adata.obsm["projections"]).copy(),
        "tensor": fit["tensor"].copy(),
        "cp_factors": [f.copy() for f in fit["cp_factors"]],
    }

    # Scramble both stages' numbering the way a separate fit could.
    rise_p, cp_p = np.arange(RISE_RANK)[::-1], np.arange(CP_RANK)[::-1]
    adata.uns["Pf2_A"] = adata.uns["Pf2_A"][:, rise_p]
    adata.uns["Pf2_B"] = adata.uns["Pf2_B"][np.ix_(rise_p, rise_p)]
    adata.uns["Pf2_weights"] = adata.uns["Pf2_weights"][rise_p]
    adata.varm["Pf2_C"] = np.asarray(adata.varm["Pf2_C"])[:, rise_p]
    adata.obsm["projections"] = np.asarray(adata.obsm["projections"])[:, rise_p]
    fit["tensor"] = fit["tensor"][:, rise_p][:, :, rise_p]
    A, B, C, D = fit["cp_factors"]
    fit["cp_factors"] = [f[:, cp_p] for f in (A, B[rise_p], C[rise_p], D)]
    fit["cp_weights"] = fit["cp_weights"][cp_p]

    relabel_workflow_fit(fit, core, "sample")

    np.testing.assert_allclose(adata.uns["Pf2_A"], expected["Pf2_A"])
    np.testing.assert_allclose(adata.obsm["projections"], expected["projections"])
    np.testing.assert_allclose(fit["tensor"], expected["tensor"])
    for got, want in zip(fit["cp_factors"], expected["cp_factors"], strict=True):
        np.testing.assert_allclose(got, want)


# =============================================================================
# Matched modes
# =============================================================================


def test_axis_skip_modes_holds_out_eigenstate_and_jackknifed_modes():
    assert axis_skip_modes(STAGE_RISE, SEED_AXIS) == (1,)
    assert axis_skip_modes(STAGE_RISE, JACKKNIFE_AXIS) == (0, 1)
    assert axis_skip_modes(STAGE_CPD, SEED_AXIS) == (1, 2)
    assert axis_skip_modes(STAGE_CPD, JACKKNIFE_AXIS) == (0, 1, 2)


def test_stability_replicates_compares_only_unperturbed_modes():
    adata = _simulate_adata(n_cond=6)
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=(SEED_AXIS, JACKKNIFE_AXIS),
        n_replicates=1,
    )

    for key, rows in df.groupby(["Axis", "Stage"]):
        axis, stage = cast(tuple[str, str], key)
        skipped = axis_skip_modes(stage, axis)
        for mode in range(4):
            compared = rows[f"SignedCosMode{mode}"].notna().all()
            assert compared == (
                mode not in skipped and (stage == STAGE_CPD or mode < 3)
            )
    # The jackknife's ANOVA is aligned by the LR mode alone.
    cpd = df[(df["Axis"] == JACKKNIFE_AXIS) & (df["Stage"] == STAGE_CPD)]
    np.testing.assert_allclose(cpd["FMS"], cpd["SignedCosMode3"].abs())


# =============================================================================
# The replicate loop
# =============================================================================


def test_stability_replicates_emits_one_row_per_axis_stage_and_component():
    adata = _simulate_adata(n_cond=6)
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=(SEED_AXIS, JACKKNIFE_AXIS),
        n_replicates=2,
        random_state=0,
    )

    counts = df.groupby(["Axis", "Stage"]).size()
    # The jackknife runs one replicate per patient, whatever n_replicates says.
    assert counts[(JACKKNIFE_AXIS, STAGE_RISE)] == 6 * RISE_RANK
    assert counts[(JACKKNIFE_AXIS, STAGE_CPD)] == 6 * CP_RANK
    assert counts[(SEED_AXIS, STAGE_RISE)] == 2 * RISE_RANK
    assert counts[(SEED_AXIS, STAGE_CPD)] == 2 * CP_RANK

    assert df["FMS"].between(0.0, 1.0 + 1e-9).all()


def test_stability_replicates_shares_one_seed_between_a_replicates_two_stages():
    """The point of the shared loop: a replicate's RISE and CPD rows have to
    describe the same fit, not two independent ones."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=(SEED_AXIS,),
        n_replicates=2,
        random_state=7,
    )

    per_replicate = df.groupby("Replicate")["Seed"].nunique()
    assert (per_replicate == 1).all()
    assert sorted(df["Seed"].unique()) == [7 * 1000 + 1, 7 * 1000 + 2]


def test_stability_replicates_is_reproducible_from_its_random_state():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    run_kwargs = {"axes": (SEED_AXIS,), "n_replicates": 1, "random_state": 0}
    a = stability_replicates(adata, reference, lr_pairs, **kwargs, **run_kwargs)
    b = stability_replicates(adata, reference, lr_pairs, **kwargs, **run_kwargs)

    pd.testing.assert_frame_equal(a, b)


def test_stability_replicates_carries_the_per_replicate_fit_diagnostics():
    """A wide FMS spread traceable to a poor or degenerate fit has to be
    distinguishable from genuine component instability."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
        adata, reference, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
    )

    cpd = df[df["Stage"] == STAGE_CPD]
    assert cpd["R2X"].notna().all()
    assert cpd["MaxModesCollinear"].notna().all()
    # The diagnostics are replicate-level scalars repeated on each row.
    assert cpd["MaxModesCollinear"].nunique() == 1
    # RISE rows carry their own R2X but no CP-specific diagnostics.
    rise = df[df["Stage"] == STAGE_RISE]
    assert rise["R2X"].notna().all()
    assert rise["MaxModesCollinear"].isna().all()


# =============================================================================
# Devices, precision and RISE compression
# =============================================================================


def test_fit_cp_reports_its_own_fit():
    """`fit_cp` returns factors on the tensor's own modes, their R2X against
    the tensor, and their collinearity, whatever device and precision it
    fits in."""
    rng = np.random.default_rng(21)
    tensor = np.abs(rng.random((20, 9, 9, 30)))

    for dtype in ("float32", "float64"):
        weights, factors, r2x, diagnostics = fit_cp(
            tensor,
            rank=3,
            n_iter_max=200,
            tol=1e-8,
            random_state=0,
            device="cpu",
            dtype=dtype,
        )
        for f, dim in zip(factors, tensor.shape, strict=True):
            assert f.shape == (dim, 3)
        assert r2x == pytest.approx(calculate_r2x(weights, factors, tensor), abs=1e-6)

        from cccrise.rank_selection import cpd_component_collinearity

        recomputed = cpd_component_collinearity({3: (weights, factors)}).iloc[0]
        assert diagnostics["MaxModesCollinear"] == int(recomputed["MaxModesCollinear"])


def test_fit_stability_pipeline_runs_compressed():
    """`compress=True` (RISE only), what the runners pass, must be a working
    code path."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()

    fit = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, compress=True
    )
    assert 0.0 <= fit["r2x_rise"] <= 1.0 + 1e-9
    assert 0.0 <= fit["r2x_cp"] <= 1.0 + 1e-9


def test_fit_stability_pipeline_accepts_an_explicit_cpu_backend():
    """`backend` must reach `fit_rise` without erroring on a plain CPU
    box, which is what every CI run and every cohort-scale fit uses."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()

    fit = fit_stability_pipeline(
        adata.copy(), lr_pairs, **kwargs, random_state=0, backend="cpu"
    )
    assert 0.0 <= fit["r2x_rise"] <= 1.0 + 1e-9


def test_stability_replicates_runs_compressed():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(
        adata.copy(),
        lr_pairs,
        **kwargs,
        random_state=0,
        compress=True,
    )

    df = stability_replicates(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=(SEED_AXIS,),
        n_replicates=1,
        compress=True,
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
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
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
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=1)

    df = stability_replicates(
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
    `parafac2.store_pf2` writes `uns`/`varm`, so the seed axis has to fit a
    copy. Without the copy the caller's object would carry the first
    replicate's fit into every later replicate."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    stability_replicates(
        adata, reference, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
    )

    assert "Pf2_A" not in adata.uns
    assert "condition_unique_idxs" not in adata.obs.columns


def test_stability_replicates_rejects_an_unknown_axis():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    with pytest.raises(ValueError, match="Unknown stability axes"):
        stability_replicates(
            adata, reference, lr_pairs, **kwargs, axes=("genes",), n_replicates=1
        )

    # The retired bootstrap axes are unknown now, not silently accepted: a
    # stale command line should fail rather than run something else.
    for retired in ("cells", "conditions", "lr_pairs", "permuted"):
        with pytest.raises(ValueError, match="Unknown stability axes"):
            stability_replicates(
                adata, reference, lr_pairs, **kwargs, axes=(retired,), n_replicates=1
            )


def test_stability_replicates_rejects_a_replicate_with_a_different_lr_roster():
    """A silent roster difference is the dangerous case: matching would line
    up different interactions row by row and return a plausible number."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)
    shortened = {**reference, "lr_symbols": reference["lr_symbols"][:-1]}

    with pytest.raises(ValueError, match="different ligand-receptor"):
        stability_replicates(
            adata, shortened, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
        )


def test_stability_replicates_rejects_a_replicate_with_a_different_gene_roster():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)
    shortened = {**reference, "gene_names": reference["gene_names"][:-1]}

    with pytest.raises(ValueError, match="different genes"):
        stability_replicates(
            adata, shortened, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
        )


def test_stability_replicates_holds_out_expected_modes():
    """RISE holds out Mode 1 (B basis); CPD holds out Modes 1 and 2 (D, E)."""
    assert RISE_SKIP_MODES == (1,)
    assert CPD_SKIP_MODES == (1, 2)

    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
        adata, reference, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
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
        {"Axis": axis, "Replicate": b, "Rank": rank, "Fold": fold, "M": value}
        for axis, spread in [(SEED_AXIS, 0.01), (JACKKNIFE_AXIS, 0.10)]
        for b, offset in enumerate([-1.0, 0.0, 1.0])
        for rank in (5, 10)
        for fold, value in enumerate([rank + spread * offset] * 2)
    ]
    summary = metric_interval(pd.DataFrame(rows), "M", by=("Axis", "Rank")).set_index(
        ["Axis", "Rank"]
    )

    seed = summary.loc[(SEED_AXIS, 5)]
    jackknife = summary.loc[(JACKKNIFE_AXIS, 5)]
    assert seed["NReplicates"] == 3
    assert seed["Median"] == 5.0
    seed_band = seed["Max"] - seed["Min"]
    assert jackknife["Max"] - jackknife["Min"] > seed_band
    assert seed_band < 5.0


def test_metric_interval_groups_by_the_keys_it_is_given():
    """Grouped per component the fold-mean is an identity; grouped per stage it
    averages components within a replicate, which is the aggregate FMS."""
    rows = [
        {
            "Axis": JACKKNIFE_AXIS,
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

    Component 1 reseeds almost perfectly but the jackknife moves it well
    below that, so the perturbation genuinely cost it something. Component 2
    reseeds worse than component 1 scores under the jackknife -- it is fuzzy at
    this rank however it is fit -- so its lower observed FMS is not the data's
    doing. A ceiling pooled across components would put the band edge at 0.45
    and call both components fine; matching per component separates them, so
    these values are what makes that behaviour testable rather than incidental.
    """
    rows = [
        {
            "Axis": JACKKNIFE_AXIS,
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
    # Component 1 reseeds better than any jackknife replicate reproduced it,
    # so leaving a patient out moved it; component 2 never drops below what reseeding alone
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
    assert (df[df["Axis"] == JACKKNIFE_AXIS]["FMS"] >= pooled_edge).all()

    summary = summarize_component_stability(df).set_index("Component")
    assert summary.loc[1, "FracWithinSeedBand"] == 0.0


def test_summarize_component_stability_reports_nan_when_no_seed_axis_was_run():
    """0.0 would report every component as having failed a test that was
    never performed."""
    summary = summarize_component_stability(_stability_frame(with_ceiling=False))

    assert summary["SeedMedianFMS"].isna().all()
    assert summary["SeedP5FMS"].isna().all()
    assert summary["FracWithinSeedBand"].isna().all()


def test_summarize_component_stability_survives_a_parquet_round_trip(tmp_path):
    """The stability table summarizes identically after being written to and
    read back from the runner's storage format."""
    path = tmp_path / "stability.parquet"
    _stability_frame(with_ceiling=True).to_parquet(path, index=False)

    summary = summarize_component_stability(pd.read_parquet(path))

    assert summary["SeedP5FMS"].notna().all()
    assert summary["FracWithinSeedBand"].notna().all()


def test_fit_stability_pipeline_labels_conditions_in_row_order():
    """`condition_labels` must name mode-0 row for row.

    `add_cond_idxs` orders the rows by a sort in the condition column's own
    dtype, so an integer column puts row 2 at patient ``10``. Deriving the
    labels with an independent string sort would instead call that row ``2``,
    and the core-numbering check -- which reads this array as positionally
    aligned to the rows -- would then compare the wrong patients without
    raising.
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


# =============================================================================
# Component--phenotype association
# =============================================================================


def test_fit_stability_pipeline_runs_the_anova_only_when_given_a_group():
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()

    fit = fit_stability_pipeline(
        adata.copy(), lr_pairs, **_pipeline_kwargs(), random_state=0
    )
    assert list(fit["anova"]["Component"]) == list(range(1, CP_RANK + 1))
    assert set(ANOVA_COLUMNS).issubset(fit["anova"].columns)
    assert fit["anova_data"].n_obs == adata.n_obs

    no_group = fit_stability_pipeline(
        adata.copy(), lr_pairs, **_pipeline_kwargs(group_column=None), random_state=0
    )
    assert no_group["anova"] is None and no_group["anova_data"] is None


def test_stability_replicates_carries_the_anova_onto_cpd_rows_only():
    adata = _simulate_adata(n_cond=6)
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=(SEED_AXIS, JACKKNIFE_AXIS),
        n_replicates=1,
    )

    cpd = df[df["Stage"] == STAGE_CPD]
    assert set(cpd["Axis"]) == {SEED_AXIS, JACKKNIFE_AXIS}
    assert cpd["EtaSquared"].between(0.0, 1.0 + 1e-9).all()
    assert cpd["P"].notna().all()
    assert (cpd["PAdj"] >= cpd["P"]).all()
    assert df.loc[df["Stage"] == STAGE_RISE, list(ANOVA_COLUMNS)].isna().all().all()


def test_seed_axis_anova_reproduces_the_reference_when_it_draws_the_reference_seed():
    """The replicate's ANOVA is the reference's when the fit is the same,
    so the match carries each statistic back onto the right component."""
    adata = _simulate_adata()
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs(control_label="ctrl", other_label="case")
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=1)

    df = stability_replicates(
        adata, reference, lr_pairs, **kwargs, axes=(SEED_AXIS,), n_replicates=1
    )

    cpd = df[df["Stage"] == STAGE_CPD].set_index("Component")
    ref = reference["anova"].set_index("Component")
    np.testing.assert_allclose(cpd["EtaSquared"], ref["EtaSquared"], rtol=1e-4)
    np.testing.assert_allclose(cpd["GroupMeanDiff"], ref["GroupMeanDiff"], rtol=1e-4)


def test_attach_anova_reads_the_matched_component_and_corrects_the_sign():
    block = pd.DataFrame(
        {
            "Component": [1, 2],
            "MatchedComponent": [2, 1],
            "SignedCosMode0": [-0.9, np.nan],
            "SignedCosMode3": [0.5, -0.7],
        }
    )
    anova = pd.DataFrame(
        {
            "Component": [1, 2],
            "EtaSquared": [0.1, 0.5],
            "F": [1.0, 5.0],
            "P": [0.4, 0.01],
            "PAdj": [0.4, 0.02],
            "GroupMeanDiff": [-1.0, 3.0],
        }
    )

    out = _attach_anova(block, anova)

    np.testing.assert_allclose(out["EtaSquared"], [0.5, 0.1])
    np.testing.assert_allclose(out["P"], [0.01, 0.4])
    np.testing.assert_allclose(out["PAdj"], [0.02, 0.4])
    # Reference component 1 is replicate component 2, flipped in the condition
    # mode; component 2's condition mode was held out, so the LR mode's sign
    # corrects it.
    np.testing.assert_allclose(out["GroupMeanDiff"], [-3.0, 1.0])


# =============================================================================
# Patient jackknife
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


def test_jackknife_units_are_sorted_patients():
    adata = _simulate_adata_with_paired_patient()
    np.testing.assert_array_equal(
        jackknife_units(adata, "patient"), ["P2", "P3", "P4", "P5", "P_pair"]
    )


def test_leave_out_patient_drops_every_condition_of_the_patient():
    adata = _simulate_adata_with_paired_patient()
    adata.obs["sample"] = adata.obs["sample"].astype("category")

    left = leave_out_patient(adata, "patient", "P_pair")

    assert set(left.obs["sample"]) == {"S2", "S3", "S4", "S5"}
    assert left.n_obs == adata.n_obs - 2 * 8
    np.testing.assert_allclose(
        left.var["means"], np.asarray(cast(Any, left.X).mean(axis=0)).ravel(), rtol=1e-6
    )
    # The caller's object is untouched.
    assert adata.n_obs == 6 * 8


def test_leave_out_patient_rejects_an_unknown_patient():
    with pytest.raises(ValueError, match="No cells"):
        leave_out_patient(_simulate_adata(), "sample", "nobody")


def test_jackknife_and_seed_are_the_only_axes():
    assert DEFAULT_AXES == (SEED_AXIS, JACKKNIFE_AXIS)


def test_jackknife_axis_leaves_out_each_patient_once():
    """One replicate per patient whatever `n_replicates` says, each a refit on
    the cohort without that patient, with the ANOVA re-run on it."""
    adata = _simulate_adata(n_cond=6)
    adata.obs["patient"] = adata.obs["sample"].replace({"S1": "S0"})
    lr_pairs = _lr_pairs()
    kwargs = _pipeline_kwargs()
    reference = fit_stability_pipeline(adata.copy(), lr_pairs, **kwargs, random_state=0)

    df = stability_replicates(
        adata,
        reference,
        lr_pairs,
        **kwargs,
        axes=(JACKKNIFE_AXIS, SEED_AXIS),
        n_replicates=1,
        patient_column="patient",
    )

    jk = df[df["Axis"] == JACKKNIFE_AXIS]
    assert jk["Replicate"].nunique() == 5
    left_out = jk.groupby("Replicate")["LeftOut"].first()
    assert list(left_out) == ["S0", "S2", "S3", "S4", "S5"]
    n_conditions = jk.groupby("LeftOut")["NConditions"].first()
    assert n_conditions["S0"] == 4  # S0 and S1 are one patient
    assert n_conditions["S3"] == 5
    assert set(jk["Stage"]) == {STAGE_RISE, STAGE_CPD}
    assert jk["FMS"].between(0.0, 1.0 + 1e-9).all()
    # Leaving out the paired patient leaves one 'ctrl' condition, which stays
    # in the ANOVA as a group of one, so that replicate still gets statistics.
    assert jk.loc[jk["Stage"] == STAGE_CPD, "EtaSquared"].notna().all()
    assert df.loc[df["Axis"] == SEED_AXIS, "LeftOut"].isna().all()


def test_jackknife_interval_scales_the_leave_one_out_spread():
    eta = np.array([0.2, 0.3, 0.4, np.nan])
    se = np.sqrt(2 / 3 * 0.02)
    low, high = _eta_squared_interval(eta, 0.3, jackknife=True, ci_level=95.0)
    np.testing.assert_allclose([low, high], [0.3 - 1.959964 * se, 0.3 + 1.959964 * se])
    # Percentiles of the same values would be far narrower.
    p_low, p_high = _eta_squared_interval(eta, 0.3, jackknife=False, ci_level=95.0)
    assert high - low > p_high - p_low


# =============================================================================
# Unstable-component selection (stability supplement)
# =============================================================================


def test_bimodal_components_flags_a_second_population_but_not_outliers():
    from cccrise.figures.figure2.figureS_cmp_stability import _common as S

    rng = np.random.default_rng(0)
    tight = 1 - rng.uniform(0, 0.01, 40)
    one_low = np.r_[np.ones(39), 0.1]  # one failed replicate is not a mode
    bimodal = np.r_[rng.uniform(0.0, 0.2, 10), rng.uniform(0.95, 1.0, 30)]
    high_bimodal = np.r_[rng.uniform(0.55, 0.65, 20), rng.uniform(0.95, 1.0, 20)]
    df = pd.concat(
        [
            pd.DataFrame(
                {"Stage": stage, "Axis": JACKKNIFE_AXIS, "Component": c, "FMS": fms}
            )
            for stage, c, fms in [
                (STAGE_RISE, 1, tight),
                (STAGE_RISE, 2, one_low),
                (STAGE_RISE, 3, high_bimodal),
                (STAGE_CPD, 1, bimodal),
            ]
        ]
    )

    assert S.bimodal_components(df) == [(STAGE_RISE, 3), (STAGE_CPD, 1)]
