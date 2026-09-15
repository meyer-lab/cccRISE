import anndata
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

import cccrise.logreg
from cccrise.figures.commonFuncs.plotGeneral import get_sample_to_group
from cccrise.import_data import add_cond_idxs
from cccrise.stability.logreg_stability import (
    align_to_reference_components,
    bootstrap_logreg_weights,
    bootstrap_replicate_weights,
    fit_logreg_pipeline,
    summarize_bootstrap_weights,
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


def test_replicate_seed_determinism():
    assert replicate_seed(0, 0) == 1
    assert replicate_seed(0, 5) == 6
    assert replicate_seed(42, 3) == 42004


def test_fit_logreg_pipeline_output_structure():
    adata = _synthetic_cohort()
    lr_pairs = pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_2", "Gene_4"],
            "receptor": ["Gene_1", "Gene_3", "Gene_5"],
            "interaction_symbol": ["Gene_0_Gene_1", "Gene_2_Gene_3", "Gene_4_Gene_5"],
        }
    )
    fit = fit_logreg_pipeline(
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
        "logreg_weights",
        "r2x_cp",
        "sample_to_group",
    }
    assert expected_keys.issubset(fit.keys())
    assert len(fit["factors"]) == 4
    assert len(fit["logreg_weights"]) == 2
    assert len(fit["lr_symbols"]) == 3
    assert isinstance(fit["r2x_cp"], float)
    assert len(fit["sample_to_group"]) == 20


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

    # Suppose raw replicate weight was -3.0 on comp 2
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


def test_summarize_bootstrap_weights_metrics():
    # Component 1: positive, stable, excludes zero
    # Component 2: centered at 0, frequently regularized to 0
    # Component 3: negative reference, stable negative
    boot_rows = []
    for b in range(100):
        boot_rows.append(
            {
                "Replicate": b,
                "Component": 1,
                "MatchedComponent": 1,
                "Weight": 2.0 + 0.1 * (b % 5 - 2),
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
                "Weight": 0.0 if b < 80 else 0.5,
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
                "Weight": -1.5 + 0.05 * (b % 3 - 1),
                "MatchScore": 0.80,
                "SignedCosine": 0.80,
                "RunnerUpScore": 0.3,
                "R2XCP": 0.85,
            }
        )

    boot_df = pd.DataFrame(boot_rows)
    reference_weights = np.array([2.0, 0.0, -1.5])
    summary = summarize_bootstrap_weights(boot_df, reference_weights)

    assert len(summary) == 3
    s1 = summary[summary["Component"] == 1].iloc[0]
    assert s1["CIExcludesZero"]
    assert s1["FracNonzero"] == 1.0
    assert s1["FracSameSignAsReference"] == 1.0
    assert s1["CILow"] > 0

    s2 = summary[summary["Component"] == 2].iloc[0]
    assert not s2["CIExcludesZero"]
    assert s2["FracNonzero"] == 0.2
    assert np.isnan(s2["FracSameSignAsReference"])  # Reference is 0.0

    s3 = summary[summary["Component"] == 3].iloc[0]
    assert s3["CIExcludesZero"]
    assert s3["FracSameSignAsReference"] == 1.0
    assert s3["CIHigh"] < 0


def test_bootstrap_replicate_weights_mismatched_lr_roster():
    factors = _synthetic_factors(n_lr=5, rank=2)
    ref = {
        "factors": factors,
        "cp_weights": np.array([1.0, 1.0]),
        "lr_symbols": np.array(["L1_R1", "L2_R2"]),
        "logreg_weights": np.array([1.0, -1.0]),
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
        bootstrap_replicate_weights(
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
    """Verify synthetic AnnData runs through fit_logreg_pipeline, bootstrap_logreg_weights,
    and summarize_bootstrap_weights without error."""
    adata = _synthetic_cohort()

    lr_pairs = pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_2", "Gene_4"],
            "receptor": ["Gene_1", "Gene_3", "Gene_5"],
            "interaction_symbol": ["Gene_0_Gene_1", "Gene_2_Gene_3", "Gene_4_Gene_5"],
        }
    )

    reference = fit_logreg_pipeline(
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
    assert len(reference["logreg_weights"]) == 2
    assert len(reference["lr_symbols"]) == 3

    boot_df = bootstrap_logreg_weights(
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
    assert set(boot_df.columns) == {
        "Replicate",
        "Component",
        "MatchedComponent",
        "Weight",
        "MatchScore",
        "SignedCosine",
        "RunnerUpScore",
        "R2XCP",
    }

    summary = summarize_bootstrap_weights(boot_df, reference["logreg_weights"])
    assert len(summary) == 2
    assert list(summary["Component"]) == [1, 2]
    assert (summary["NReplicates"] == 2).all()


def test_get_sample_to_group_is_ordered_by_condition_unique_idxs():
    """`fit_logreg_weights` reads `sample_to_group` as aligned to `uns["A"]`'s
    rows, which are in `condition_unique_idxs` order. The ``__rep2`` suffix a
    repeat draw carries sorts next to its source label, but the alignment must
    not rest on that -- it rests on both orderings being the same sort."""
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


def test_fit_logreg_weights_hands_the_splitter_its_groups(monkeypatch):
    """No copy of a bootstrapped patient may straddle a split.

    Ungrouped folds score a held-out half against rows the fit has already
    seen, and select the L1 strength off that optimistic score. `groups`
    reaches the splitter only through scikit-learn's metadata routing, so the
    check is on the folds the fit actually drew: a routing failure would fall
    back to ungrouped folds silently.
    """
    n_source = 20
    labels = [f"Sample_{i}" for i in range(n_source)]
    # Four patients drawn twice, as a bootstrapped roster would have them.
    labels += [f"Sample_{i}__rep2" for i in (0, 1, 10, 11)]
    sources = np.array([label.split("__rep")[0] for label in labels])
    y = pd.Series(
        [0 if int(source.split("_")[1]) < n_source // 2 else 1 for source in sources],
        index=labels,
    )
    rng = np.random.default_rng(0)
    X = anndata.AnnData(
        X=np.zeros((len(labels), 1)),
        uns={"A": rng.standard_normal((len(labels), 3))},
    )

    drawn: list[tuple[np.ndarray, np.ndarray]] = []

    class RecordingSplitter(StratifiedGroupKFold):
        def split(self, X, y=None, groups=None):
            groups = np.asarray(groups)
            for train, test in super().split(X, y, groups):
                drawn.append((groups[train], groups[test]))
                yield train, test

    monkeypatch.setattr(cccrise.logreg, "StratifiedGroupKFold", RecordingSplitter)
    cccrise.logreg.fit_logreg_weights(X, y, groups=list(sources))

    assert drawn
    for train_sources, test_sources in drawn:
        assert not (set(train_sources) & set(test_sources))
