"""The ALAD comparator bootstrap's resampling helpers
(`comparators.resampling`), moved with their tests from the former cccRISE
bootstrap pipeline."""

from typing import cast

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from cccrise.comparators.resampling import (
    _carry_paired_samples,
    _paired_conditions,
    resample_conditions,
)


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


def test_carry_paired_samples_keeps_both_of_the_pair_when_it_can():
    """With other draws to drop, the compensating drop never undoes the
    carry: the trigger and its carried-in sibling both survive."""
    pair_of = {"S0": "S1", "S1": "S0"}
    drawn = ["S0", "S2", "S3", "S4"]
    for seed in range(30):
        rng = np.random.default_rng(seed)
        result = _carry_paired_samples(drawn, pair_of, rng)
        assert len(result) == len(drawn)
        assert {"S0", "S1"} <= set(result)


def test_carry_paired_samples_skips_draws_an_earlier_carry_dropped():
    """A draw removed by an earlier carry's drop must not carry its own
    sibling in afterwards."""
    pair_of = {"S0": "S1", "S1": "S0", "S2": "S3", "S3": "S2"}
    # Enough unpaired draws that no drop ever has to fall back on a pair
    drawn = ["S0", "S2", "S4", "S5", "S6"]
    dropped_s2 = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        result = _carry_paired_samples(drawn, pair_of, rng)
        assert len(result) == len(drawn)
        if "S2" not in result:
            dropped_s2 += 1
            assert "S3" not in result
        else:
            assert "S3" in result
    assert dropped_s2 > 0


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
