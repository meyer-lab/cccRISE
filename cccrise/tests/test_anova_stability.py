from typing import Any

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from cccrise.anova import (
    _eta_squared,
    adjust_pvalues,
    anova_component_associations,
    prepare_group_labels,
    run_multiple_factors_anova,
    run_saved_factors_anova,
)
from cccrise.figures.commonFuncs.plotGeneral import get_sample_to_group
from cccrise.import_data import add_cond_idxs
from cccrise.runners._cli import CP_TOL, RISE_TOL
from cccrise.utils import replicate_seed, resample_conditions
from cccrise.validation.anova_stability import (
    align_to_reference_components,
    bootstrap_anova,
    bootstrap_replicate_anova,
    fit_anova_pipeline,
    summarize_bootstrap_anova,
    top_lr_pairs,
)


def _synthetic_factors(
    n_lr: int = 10, rank: int = 3, seed: int = 42
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((8, rank))
    B = rng.standard_normal((5, rank))
    C = rng.standard_normal((5, rank))
    D = rng.standard_normal((n_lr, rank))
    return [A, B, C, D]


def _synthetic_cohort(
    n_conditions: int = 20, cells_per_cond: int = 10, n_genes: int = 8, seed: int = 123
) -> anndata.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.exponential(scale=1.0, size=(n_conditions * cells_per_cond, n_genes))
    samples = [
        f"Sample_{i}" for i in range(n_conditions) for _ in range(cells_per_cond)
    ]
    groups = [
        "Ctrl" if i < n_conditions // 2 else "Treat"
        for i in range(n_conditions)
        for _ in range(cells_per_cond)
    ]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"Sample": samples, "Group": groups}),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var["means"] = np.asarray(X.mean(axis=0)).ravel()
    return adata


# =============================================================================
# Determinism and alignment helpers
# =============================================================================


def test_replicate_seed_determinism():
    assert replicate_seed(0, 0) == 1
    assert replicate_seed(0, 5) == 6
    assert replicate_seed(42, 3) == 42004


def test_align_to_reference_components_exact_match():
    factors = _synthetic_factors(n_lr=10, rank=3, seed=1)
    values = np.array([1.5, -2.0, 0.5])
    aligned = align_to_reference_components(factors, factors, values)

    assert len(aligned) == 3
    assert list(aligned["Component"]) == [1, 2, 3]
    assert list(aligned["MatchedComponent"]) == [1, 2, 3]
    np.testing.assert_allclose(aligned["Value"], values)
    np.testing.assert_allclose(aligned["MatchScore"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(aligned["SignedCosine"], [1.0, 1.0, 1.0])


def test_align_to_reference_components_sign_flip():
    ref_factors = _synthetic_factors(n_lr=10, rank=2, seed=2)
    rep_factors = [f.copy() for f in ref_factors]
    # Invert sign of component 2 in replicate Mode 3 (LR)
    rep_factors[3][:, 1] = -rep_factors[3][:, 1]

    # Suppose raw replicate difference was -3.0 on comp 2
    raw_values = np.array([1.0, -3.0])
    aligned = align_to_reference_components(ref_factors, rep_factors, raw_values)

    assert aligned.loc[aligned["Component"] == 2, "MatchedComponent"].iloc[0] == 2
    np.testing.assert_allclose(
        aligned.loc[aligned["Component"] == 2, "SignedCosine"].iloc[0], -1.0
    )
    # Value should be flipped from -3.0 to +3.0 to match reference orientation
    np.testing.assert_allclose(
        aligned.loc[aligned["Component"] == 2, "Value"].iloc[0], 3.0
    )


def test_align_to_reference_components_permutation():
    ref_factors = _synthetic_factors(n_lr=10, rank=3, seed=3)
    rep_factors = [f.copy() for f in ref_factors]
    # Swap columns 0 and 2 in all replicate modes
    for m in range(4):
        rep_factors[m][:, [0, 2]] = rep_factors[m][:, [2, 0]]

    raw_values = np.array([10.0, 20.0, 30.0])  # raw values on comp 1, 2, 3 of replicate
    aligned = align_to_reference_components(ref_factors, rep_factors, raw_values)

    # Reference comp 1 was moved to replicate comp 3 (raw value 30.0)
    # Reference comp 2 stayed replicate comp 2 (raw value 20.0)
    # Reference comp 3 was moved to replicate comp 1 (raw value 10.0)
    comp1_row = aligned[aligned["Component"] == 1].iloc[0]
    comp3_row = aligned[aligned["Component"] == 3].iloc[0]

    assert comp1_row["MatchedComponent"] == 3
    np.testing.assert_allclose(comp1_row["Value"], 30.0)
    assert comp3_row["MatchedComponent"] == 1
    np.testing.assert_allclose(comp3_row["Value"], 10.0)


def test_top_lr_pairs():
    lr_symbols = np.array(["L1_R1", "L2_R2", "L3_R3", "L4_R4", "L5_R5"])
    D = np.array(
        [
            [0.1, 0.9],
            [0.8, -0.2],
            [0.3, 0.4],
            [-0.95, 0.05],
            [0.0, -0.85],
        ]
    )
    df = top_lr_pairs(D, lr_symbols, top_n=2)
    assert len(df) == 4  # 2 components * 2 slots

    c1 = df[df["Component"] == 1].sort_values("TopRank")
    assert list(c1["LRPair"]) == ["L4_R4", "L2_R2"]
    np.testing.assert_allclose(c1["DLoading"], [-0.95, 0.8])

    c2 = df[df["Component"] == 2].sort_values("TopRank")
    assert list(c2["LRPair"]) == ["L1_R1", "L5_R5"]


# =============================================================================
# ANOVA mathematics and degenerate guards
# =============================================================================


def test_eta_squared_matches_hand_computed_sums_of_squares():
    """Verify eta-squared SS_between / SS_total matches manual formula."""
    group1 = np.array([2.0, 4.0, 6.0])
    group2 = np.array([10.0, 12.0, 14.0])
    # Grand mean = 8.0, SS_between = 96.0, SS_within = 16.0, SS_total = 112.0
    expected_eta = 96.0 / 112.0
    computed_eta = _eta_squared([group1, group2])
    np.testing.assert_allclose(computed_eta, expected_eta)


def test_two_group_anova_f_equals_squared_t():
    """Verify one-way ANOVA F-statistic on two groups equals two-sample t^2."""
    rng = np.random.default_rng(42)
    n1, n2 = 12, 15
    g1 = rng.normal(loc=1.0, scale=1.5, size=n1)
    g2 = rng.normal(loc=2.5, scale=1.5, size=n2)

    A = np.column_stack([np.concatenate([g1, g2])])  # 1 component
    labels = pd.Series(["Ctrl"] * n1 + ["Treat"] * n2)

    adata = anndata.AnnData(X=np.zeros((len(labels), 1)), uns={"A": A})
    res = anova_component_associations(adata, labels, control_label="Ctrl")

    f_stat = res.loc[0, "F"]
    t_stat = ttest_ind(g2, g1, equal_var=True).statistic
    np.testing.assert_allclose(f_stat, t_stat**2)
    np.testing.assert_allclose(
        res.loc[0, "GroupMeanDiff"], float(g2.mean() - g1.mean())
    )


def test_anova_reports_nan_for_a_degenerate_component():
    """A component with a group having fewer than 2 samples reports NaN rather
    than raising an error, so a bootstrap replicate survives."""
    A = np.array([[1.0], [2.0], [3.0], [4.0]])
    labels = pd.Series(["Ctrl", "Treat", "Treat", "Treat"])  # Ctrl has only 1 sample

    adata = anndata.AnnData(X=np.zeros((4, 1)), uns={"A": A})
    res = anova_component_associations(adata, labels, control_label="Ctrl")

    assert len(res) == 1
    assert np.isnan(res.loc[0, "F"])
    assert np.isnan(res.loc[0, "P"])
    assert np.isnan(res.loc[0, "EtaSquared"])
    assert np.isnan(res.loc[0, "GroupMeanDiff"])
    assert not res.loc[0, "Significant"]


# =============================================================================
# Pipeline, replicate, and summary tests
# =============================================================================


def test_fit_anova_pipeline_output_structure():
    adata = _synthetic_cohort()
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_2", "Gene_4"],
            "receptor": ["Gene_1", "Gene_3", "Gene_5"],
            "interaction_symbol": ["Gene_0_Gene_1", "Gene_2_Gene_3", "Gene_4_Gene_5"],
        }
    )
    fit = fit_anova_pipeline(
        adata,
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
        rise_rank=2,
        cp_rank=2,
        lr_pairs=lr_pairs,
        rise_n_iter_max=10,
        cp_n_iter_max=10,
        rise_tol=RISE_TOL,
        cp_tol=CP_TOL,
        random_state=0,
    )

    assert isinstance(fit, dict)
    expected_keys = {
        "factors",
        "cp_weights",
        "lr_symbols",
        "anova",
        "r2x_cp",
        "sample_to_group",
    }
    assert expected_keys.issubset(fit.keys())
    assert len(fit["factors"]) == 4
    assert isinstance(fit["anova"], pd.DataFrame)
    assert len(fit["anova"]) == 2
    assert list(fit["anova"]["Component"]) == [1, 2]
    expected_anova_cols = {
        "Component",
        "F",
        "P",
        "EtaSquared",
        "GroupMeanDiff",
        "Significant",
        "NGroups",
        "GroupSizes",
    }
    assert expected_anova_cols.issubset(fit["anova"].columns)
    assert len(fit["lr_symbols"]) == 3
    assert isinstance(fit["r2x_cp"], float)
    assert len(fit["sample_to_group"]) == 20


# =============================================================================
# GPU backend and CANDELINC compression
# =============================================================================


def _anova_lr_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_2", "Gene_4"],
            "receptor": ["Gene_1", "Gene_3", "Gene_5"],
            "interaction_symbol": ["Gene_0_Gene_1", "Gene_2_Gene_3", "Gene_4_Gene_5"],
        }
    )


def test_fit_anova_pipeline_runs_with_compression_disabled():
    """`compress=False` must still be a supported code path, not just the
    default's untested complement."""
    fit = fit_anova_pipeline(
        _synthetic_cohort(),
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
        rise_rank=2,
        cp_rank=2,
        lr_pairs=_anova_lr_pairs(),
        rise_n_iter_max=10,
        cp_n_iter_max=10,
        rise_tol=RISE_TOL,
        cp_tol=CP_TOL,
        random_state=0,
        compress=False,
    )
    assert 0.0 <= fit["r2x_cp"] <= 1.0 + 1e-9


def test_fit_anova_pipeline_accepts_an_explicit_cpu_backend():
    """`backend` must reach `rise_store_r2x` without erroring on a plain CPU
    box, which is what every CI run and every cohort-scale fit uses."""
    fit = fit_anova_pipeline(
        _synthetic_cohort(),
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
        rise_rank=2,
        cp_rank=2,
        lr_pairs=_anova_lr_pairs(),
        rise_n_iter_max=10,
        cp_n_iter_max=10,
        rise_tol=RISE_TOL,
        cp_tol=CP_TOL,
        random_state=0,
        backend="cpu",
    )
    assert 0.0 <= fit["r2x_cp"] <= 1.0 + 1e-9


def test_bootstrap_anova_runs_with_compression_disabled():
    adata = _synthetic_cohort()
    lr_pairs = _anova_lr_pairs()
    kwargs: dict[str, Any] = {
        "condition_column": "Sample",
        "group_column": "Group",
        "control_label": "Ctrl",
        "other_label": "Treat",
        "rise_rank": 2,
        "cp_rank": 2,
        "lr_pairs": lr_pairs,
        "rise_n_iter_max": 10,
        "cp_n_iter_max": 10,
        "rise_tol": RISE_TOL,
        "cp_tol": CP_TOL,
    }
    reference = fit_anova_pipeline(adata, **kwargs, random_state=0, compress=False)

    df = bootstrap_anova(
        adata, reference=reference, **kwargs, n_boot=1, random_state=0, compress=False
    )
    assert df["EtaSquared"].between(-1e-9, 1.0 + 1e-9).all()


def test_bootstrap_anova_forwards_patient_column_to_the_resampler(monkeypatch):
    """Paired patients must be carried together in the ANOVA bootstrap too,
    not only in `component_stability`."""
    from cccrise.validation import anova_stability

    seen = []
    real = anova_stability.resample_conditions

    def spy(*args, **kwargs):
        seen.append(kwargs.get("patient_by"))
        return real(*args, **kwargs)

    monkeypatch.setattr(anova_stability, "resample_conditions", spy)

    adata = _synthetic_cohort()
    adata.obs["Patient"] = adata.obs["Sample"].replace({"Sample_1": "Sample_0"})
    kwargs: dict[str, Any] = {
        "condition_column": "Sample",
        "group_column": "Group",
        "control_label": "Ctrl",
        "other_label": "Treat",
        "rise_rank": 2,
        "cp_rank": 2,
        "lr_pairs": _anova_lr_pairs(),
        "rise_n_iter_max": 10,
        "cp_n_iter_max": 10,
        "rise_tol": RISE_TOL,
        "cp_tol": CP_TOL,
    }
    reference = fit_anova_pipeline(adata, **kwargs, random_state=0)
    bootstrap_anova(
        adata, reference=reference, **kwargs, n_boot=1, patient_column="Patient"
    )
    assert seen == ["Patient"]


def test_summarize_bootstrap_anova_metrics():
    # Component 1: large eta, significant, positive diff
    # Component 2: small eta, p > 0.05, 0 diff
    # Component 3: medium eta, mixed p, negative diff
    boot_rows = []
    for b in range(100):
        boot_rows.append(
            {
                "Replicate": b,
                "Component": 1,
                "MatchedComponent": 1,
                "EtaSquared": 0.6 + 0.01 * (b % 5 - 2),
                "F": 15.0 + (b % 5),
                "P": 0.001 if b < 95 else 0.06,  # 95% significant (< 0.05)
                "GroupMeanDiff": 2.0 + 0.1 * (b % 5 - 2),
                "MatchScore": 0.95,
                "SignedCosine": 0.95,
                "RunnerUpScore": 0.1,
                "R2XCP": 0.85,
            }
        )
        boot_rows.append(
            {
                "Replicate": b,
                "Component": 2,
                "MatchedComponent": 2,
                "EtaSquared": 0.02 + 0.001 * (b % 5 - 2),
                "F": 0.5 + 0.1 * (b % 5),
                "P": 0.40,
                "GroupMeanDiff": 0.0 if b < 80 else 0.5,
                "MatchScore": 0.90,
                "SignedCosine": 0.90,
                "RunnerUpScore": 0.2,
                "R2XCP": 0.85,
            }
        )
        boot_rows.append(
            {
                "Replicate": b,
                "Component": 3,
                "MatchedComponent": 3,
                "EtaSquared": 0.3 + 0.02 * (b % 3 - 1),
                "F": 8.0 + (b % 3),
                "P": 0.02 if b < 60 else 0.10,  # 60% significant
                "GroupMeanDiff": -1.5 + 0.05 * (b % 3 - 1),
                "MatchScore": 0.80,
                "SignedCosine": 0.80,
                "RunnerUpScore": 0.3,
                "R2XCP": 0.85,
            }
        )

    boot_df = pd.DataFrame(boot_rows)
    reference_anova = pd.DataFrame(
        {
            "Component": [1, 2, 3],
            "EtaSquared": [0.6, 0.02, 0.3],
            "F": [15.0, 0.5, 8.0],
            "P": [0.001, 0.40, 0.02],
            "GroupMeanDiff": [2.0, 0.0, -1.5],
        }
    )
    summary = summarize_bootstrap_anova(
        boot_df, reference_anova, ci_level=90.0, alpha=0.05
    )

    assert len(summary) == 3
    s1 = summary[summary["Component"] == 1].iloc[0]
    assert s1["FracSignificant"] == 0.95
    assert s1["FracSameSignAsReference"] == 1.0
    assert s1["CILow"] > 0.5
    assert s1["ReferenceEtaSquared"] == 0.6
    assert s1["ReferenceGroupMeanDiff"] == 2.0

    s2 = summary[summary["Component"] == 2].iloc[0]
    assert s2["FracSignificant"] == 0.0
    assert np.isnan(s2["FracSameSignAsReference"])  # ReferenceGroupMeanDiff is 0.0

    s3 = summary[summary["Component"] == 3].iloc[0]
    assert s3["FracSignificant"] == 0.60
    assert s3["FracSameSignAsReference"] == 1.0
    assert s3["ReferenceGroupMeanDiff"] == -1.5


def test_bootstrap_replicate_anova_mismatched_lr_roster():
    factors = _synthetic_factors(n_lr=5, rank=2)
    ref_anova = pd.DataFrame(
        {
            "Component": [1, 2],
            "EtaSquared": [0.5, 0.2],
            "F": [10.0, 2.0],
            "P": [0.01, 0.2],
            "GroupMeanDiff": [1.0, -1.0],
        }
    )
    ref = {
        "factors": factors,
        "cp_weights": np.array([1.0, 1.0]),
        "lr_symbols": np.array(["L1_R1", "L2_R2"]),
        "anova": ref_anova,
        "r2x_cp": 0.8,
    }

    adata = _synthetic_cohort()

    # Mismatched lr_pairs with 2 pairs so CP rank 2 is valid, but symbols differ from ref
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_2"],
            "receptor": ["Gene_1", "Gene_3"],
            "interaction_symbol": ["Gene_0_Gene_1", "Gene_2_Gene_3"],
        }
    )

    with pytest.raises(ValueError, match="different ligand-receptor"):
        bootstrap_replicate_anova(
            adata,
            reference=ref,
            condition_column="Sample",
            group_column="Group",
            control_label="Ctrl",
            other_label="Treat",
            rise_rank=2,
            cp_rank=2,
            lr_pairs=lr_pairs,
            rise_n_iter_max=10,
            cp_n_iter_max=10,
            rise_tol=RISE_TOL,
            cp_tol=CP_TOL,
        )


def test_end_to_end_synthetic_bootstrap():
    """Verify synthetic AnnData runs through fit_anova_pipeline, bootstrap_anova,
    and summarize_bootstrap_anova without error."""
    adata = _synthetic_cohort()

    lr_pairs = pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_2", "Gene_4"],
            "receptor": ["Gene_1", "Gene_3", "Gene_5"],
            "interaction_symbol": ["Gene_0_Gene_1", "Gene_2_Gene_3", "Gene_4_Gene_5"],
        }
    )

    reference = fit_anova_pipeline(
        adata,
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
        rise_rank=2,
        cp_rank=2,
        lr_pairs=lr_pairs,
        rise_n_iter_max=10,
        cp_n_iter_max=10,
        rise_tol=RISE_TOL,
        cp_tol=CP_TOL,
        random_state=0,
    )
    assert len(reference["anova"]) == 2
    assert len(reference["lr_symbols"]) == 3

    boot_df = bootstrap_anova(
        adata,
        reference=reference,
        condition_column="Sample",
        group_column="Group",
        control_label="Ctrl",
        other_label="Treat",
        rise_rank=2,
        cp_rank=2,
        n_boot=2,
        lr_pairs=lr_pairs,
        rise_n_iter_max=10,
        cp_n_iter_max=10,
        rise_tol=RISE_TOL,
        cp_tol=CP_TOL,
        random_state=0,
    )
    assert len(boot_df) == 4  # 2 replicates * 2 components
    expected_cols = {
        "Replicate",
        "Component",
        "MatchedComponent",
        "EtaSquared",
        "F",
        "P",
        "GroupMeanDiff",
        "MatchScore",
        "SignedCosine",
        "RunnerUpScore",
        "R2XCP",
    }
    assert set(boot_df.columns) == expected_cols

    summary = summarize_bootstrap_anova(boot_df, reference["anova"])
    assert len(summary) == 2
    assert list(summary["Component"]) == [1, 2]
    assert (summary["NReplicates"] == 2).all()


def test_get_sample_to_group_is_ordered_by_condition_unique_idxs():
    """`anova_component_associations` reads `sample_to_group` as aligned to
    `uns["A"]`'s rows, which are in `condition_unique_idxs` order. The ``__rep2``
    suffix a repeat draw carries sorts next to its source label, but the
    alignment must not rest on that -- it rests on both orderings being the
    same sort."""
    adata = _synthetic_cohort(n_conditions=6)
    resampled = add_cond_idxs(
        resample_conditions(adata, "Sample", stratify_by="Group", random_seed=0),
        "Sample",
    )

    sample_to_group = get_sample_to_group(resampled, "Sample", "Group")

    rows = resampled.obs["condition_unique_idxs"].to_numpy()
    labels = resampled.obs["Sample"].to_numpy().astype(str)
    for row, label in zip(rows, labels, strict=True):
        assert sample_to_group.index[row] == label


def test_adjust_pvalues_methods():
    """Verify multiple test correction methods and preservation of NaNs."""
    ps = [0.01, 0.04, 0.03, float("nan")]

    # Raw / None
    np.testing.assert_allclose(
        adjust_pvalues(ps, method=None),
        [0.01, 0.04, 0.03, np.nan],
    )

    # Bonferroni: min(p * 3, 1.0)
    np.testing.assert_allclose(
        adjust_pvalues(ps, method="bonferroni"),
        [0.03, 0.12, 0.09, np.nan],
    )

    # Benjamini-Hochberg FDR
    bh = adjust_pvalues(ps, method="fdr_bh")
    assert np.isnan(bh[3])
    assert bh[0] == pytest.approx(0.03)  # 0.01 * 3 / 1
    assert bh[2] == pytest.approx(0.04)  # 0.03 * 3 / 2 = 0.045 -> monotonized to 0.04
    assert bh[1] == pytest.approx(0.04)  # 0.04 * 3 / 3 = 0.04

    # Holm-Bonferroni
    holm = adjust_pvalues(ps, method="holm")
    assert np.isnan(holm[3])
    assert holm[0] == pytest.approx(0.03)  # 0.01 * 3
    assert holm[2] == pytest.approx(0.06)  # 0.03 * 2
    assert holm[1] == pytest.approx(0.06)  # max(0.06, 0.04 * 1)


def test_anova_component_associations_with_correction():
    """`anova_component_associations` incorporates multiple testing corrections into PAdj and Significant."""
    A = np.array(
        [
            [1.0, 10.0],
            [1.1, 10.1],
            [5.0, 1.0],
            [5.2, 1.2],
        ]
    )
    adata = anndata.AnnData(
        X=np.zeros((4, 2)),
        obs=pd.DataFrame({"sample": ["S1", "S2", "S3", "S4"]}),
        uns={"A": A},
    )
    s2g = pd.Series(["Ctrl", "Ctrl", "Treat", "Treat"], index=["S1", "S2", "S3", "S4"])

    # Uncorrected
    df_raw = anova_component_associations(adata, s2g, alpha=0.05, correction=None)
    assert "PAdj" in df_raw.columns
    np.testing.assert_allclose(df_raw["P"], df_raw["PAdj"])

    # Bonferroni corrected
    df_bonf = anova_component_associations(
        adata, s2g, alpha=0.05, correction="bonferroni"
    )
    np.testing.assert_allclose(
        df_bonf["PAdj"], np.clip(df_raw["P"] * len(df_raw), 0.0, 1.0)
    )


def test_prepare_group_labels():
    """`prepare_group_labels` correctly handles levels filtering, mapping, and binary collapsing."""
    raw = pd.Series(
        ["control", "declined", "recovered", "control", "unknown"],
        index=["S1", "S2", "S3", "S4", "S5"],
    )

    # Levels subsetting
    s_levels = prepare_group_labels(raw, levels=["control", "declined"])
    assert s_levels["S1"] == "control"
    assert s_levels["S2"] == "declined"
    assert pd.isna(s_levels["S3"])
    assert pd.isna(s_levels["S5"])

    # Mapping
    s_map = prepare_group_labels(
        raw, mapping={"control": "Healthy", "declined": "Diseased"}
    )
    assert s_map["S1"] == "Healthy"
    assert s_map["S2"] == "Diseased"
    assert pd.isna(s_map["S3"])

    # Binary collapsing
    s_collapse = prepare_group_labels(raw, control_label="control", other_label="alad")
    assert s_collapse["S1"] == "control"
    assert s_collapse["S2"] == "alad"
    assert s_collapse["S3"] == "alad"


def test_run_saved_factors_anova_and_multiple_associations():
    """`run_saved_factors_anova`, `run_multiple_factors_anova`, and `figureAnova` work."""
    import os

    # `_resolve_factors_and_metadata` falls back to loading the real ALAD
    # source file for `ALADstatus` annotations not stored in the saved
    # factors dataset; that file is private and absent on CI runners.
    if not os.path.exists("/opt/IVCSC/BAL-scRNAseq.raw.h5ad"):
        pytest.skip("requires local ALAD source data")

    import matplotlib.pyplot as plt

    from cccrise.factors import load_factors_netcdf
    from cccrise.figures.figureAnova import (
        DEFAULT_BOXPLOTS,
        makeFigure,
        plot_anova_summary_heatmap,
        plot_association_effect_sizes,
        plot_association_significance,
        plot_configured_boxplots,
    )

    ds = load_factors_netcdf("cccrise/data/bal_alad_factors.nc")
    n_cond = ds.sizes["condition"]
    cond_names = [str(c) for c in ds["condition"].values]
    s2g = pd.Series(
        [
            "control" if i % 3 == 0 else "declined" if i % 3 == 1 else "recovered"
            for i in range(n_cond)
        ],
        index=cond_names,
    )

    # Test running ANOVA on bal_alad_factors.nc with kwargs
    df = run_saved_factors_anova(
        "cccrise/data/bal_alad_factors.nc",
        sample_to_group=s2g,
        group_column="ALADstatus",
        levels=["control", "declined"],
        control_label="control",
        association_name="Control vs Declined",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == ds.sizes["cp_component"]
    assert df["Association"].iloc[0] == "Control vs Declined"
    assert {
        "Component",
        "F",
        "P",
        "PAdj",
        "EtaSquared",
        "GroupMeanDiff",
        "Significant",
        "NGroups",
        "GroupSizes",
    }.issubset(df.columns)

    # Test plotting subplots
    fig, (ax0, ax1) = plt.subplots(1, 2)
    plot_association_effect_sizes(df, ax0)
    plot_association_significance(df, ax1)
    plt.close(fig)

    # Test multiple associations with global correction
    associations_config = [
        {
            "name": "Control vs. ALAD",
            "group_column": "ALADstatus",
            "control_label": "control",
            "other_label": "alad",
        },
        {
            "name": "Control vs. Declined",
            "group_column": "ALADstatus",
            "levels": ["control", "declined"],
            "control_label": "control",
        },
    ]
    multi_df = run_multiple_factors_anova(
        "cccrise/data/bal_alad_factors.nc",
        associations=associations_config,
        global_correction=True,
    )
    assert len(multi_df) == 2 * ds.sizes["cp_component"]
    assert set(multi_df["Association"]) == {"Control vs. ALAD", "Control vs. Declined"}

    # Test summary heatmap
    fig_hm, ax_hm = plt.subplots(1, 1)
    plot_anova_summary_heatmap(multi_df, ax_hm)
    plt.close(fig_hm)

    # Test plot_configured_boxplots with dictionary configuration
    fig_bp, (ax_bp0, ax_bp1) = plt.subplots(1, 2)
    plot_configured_boxplots(
        ds,
        DEFAULT_BOXPLOTS,
        [ax_bp0, ax_bp1],
        combined_anova_df=multi_df,
    )
    plt.close(fig_bp)

    # Test makeFigure with custom associations and custom boxplots
    custom_boxplots = [
        {
            "component": 1,
            "group_column": "ALADstatus",
            "levels": ["control", "declined"],
            "title": "C1: Ctrl vs Dec",
        }
    ]
    fig_custom = makeFigure(
        associations=associations_config,
        boxplots=custom_boxplots,
    )
    assert isinstance(fig_custom, plt.Figure)
    plt.close(fig_custom)

    # Test default makeFigure (runs all ALAD status types and default boxplots)
    fig_default = makeFigure()
    assert isinstance(fig_default, plt.Figure)
    plt.close(fig_default)
