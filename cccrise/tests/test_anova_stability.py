import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from cccrise.anova import (
    _eta_squared,
    anova_component_associations,
)
from cccrise.figures.commonFuncs.plotGeneral import get_sample_to_group
from cccrise.import_data import add_cond_idxs
from cccrise.stability.anova_stability import (
    align_to_reference_components,
    bootstrap_anova,
    bootstrap_replicate_anova,
    fit_anova_pipeline,
    summarize_bootstrap_anova,
    top_lr_pairs,
)
from cccrise.utils import replicate_seed, resample_conditions


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
