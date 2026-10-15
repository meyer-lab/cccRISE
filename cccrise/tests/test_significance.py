"""
Unit tests for cccrise.validation.significance and cccrise.runners.run_significance.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from cccrise.ccc_rise import fit_rise
from cccrise.factors import save_ccc_rise_fit
from cccrise.import_data import add_cond_idxs
from cccrise.runners.datasets import DatasetSpec
from cccrise.runners.run_significance import build_parser, output_paths
from cccrise.validation.significance import (
    fit_significance_pipeline,
    patient_block_permutations,
    patient_permutation_differential,
    significance_to_xarray,
    summarize_significance_lr,
    summarize_significance_overview,
)


def write_rise_reference(
    adata: anndata.AnnData,
    condition_column: str,
    rise_rank: int,
    path: Path,
    random_state: int = 0,
) -> anndata.AnnData:
    """A factors.nc holding a real RISE fit and rank-1 CP placeholders (a RISE
    reference never reads its CP portion); returns the fitted AnnData."""
    fit_adata, _ = fit_rise(
        add_cond_idxs(adata.copy(), condition_column),
        rank=rise_rank,
        n_iter_max=100,
        tolerance=1e-6,
        random_state=random_state,
        backend="cpu",
    )
    n_cond = fit_adata.obs[condition_column].nunique()
    placeholder_cp = {
        "cp_factors": [
            np.ones((n_cond, 1)),
            np.ones((rise_rank, 1)),
            np.ones((rise_rank, 1)),
            np.ones((1, 1)),
        ],
        "cp_weights": np.ones(1),
        "filtered_lr_pairs": ["placeholder"],
        "r2x_cp": 0.0,
    }
    save_ccc_rise_fit(path, fit_adata, placeholder_cp, condition_column)
    return fit_adata


def _synthetic_cohort_for_significance(
    n_conditions: int = 6,
    cells_per_cond: int = 15,
    n_genes: int = 8,
    seed: int = 42,
) -> tuple[anndata.AnnData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = rng.exponential(scale=1.0, size=(n_conditions * cells_per_cond, n_genes))
    samples = [
        f"Sample_{i}" for i in range(n_conditions) for _ in range(cells_per_cond)
    ]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"sample": samples}),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var["means"] = np.asarray(X.mean(axis=0)).ravel()

    lr_pairs = pd.DataFrame(
        {
            "ligand": ["Gene_0", "Gene_1", "Gene_2&Gene_3"],
            "receptor": ["Gene_4", "Gene_5", "Gene_6&Gene_7"],
            "interaction_symbol": ["Gene0_Gene4", "Gene1_Gene5", "Gene23_Gene67"],
        }
    )
    return adata, lr_pairs


def test_fit_significance_pipeline_end_to_end(tmp_path):
    adata, lr_pairs = _synthetic_cohort_for_significance()
    rise_rank = 3
    n_perms = 20
    reference_path = tmp_path / "reference.nc"
    write_rise_reference(adata, "sample", rise_rank, reference_path)

    results = fit_significance_pipeline(
        adata=adata,
        condition_column="sample",
        reference_path=reference_path,
        lr_pairs=lr_pairs,
        n_perms=n_perms,
        random_state=42,
        validate=True,
    )

    n_cond = 6
    n_lr = len(results["filtered_lr_pairs"])
    assert results["observed_tensor"].shape == (n_cond, rise_rank, rise_rank, n_lr)
    assert results["pvalue_tensor"].shape == (n_cond, rise_rank, rise_rank, n_lr)
    assert results["null_mean_tensor"].shape == (n_cond, rise_rank, rise_rank, n_lr)
    assert results["null_std_tensor"].shape == (n_cond, rise_rank, rise_rank, n_lr)

    # Check bounds
    assert np.all(results["pvalue_tensor"] >= 0.0)
    assert np.all(results["pvalue_tensor"] <= 1.0)
    assert np.all(results["null_std_tensor"] >= 0.0)


def test_summaries_and_xarray_conversion(tmp_path):
    adata, lr_pairs = _synthetic_cohort_for_significance()
    reference_path = tmp_path / "reference.nc"
    write_rise_reference(adata, "sample", 2, reference_path)
    results = fit_significance_pipeline(
        adata=adata,
        condition_column="sample",
        reference_path=reference_path,
        lr_pairs=lr_pairs,
        n_perms=10,
        random_state=42,
    )

    # 1. LR summary
    lr_df = summarize_significance_lr(results)
    assert isinstance(lr_df, pd.DataFrame)
    assert len(lr_df) == len(results["filtered_lr_pairs"])
    assert "MeanObserved" in lr_df.columns
    assert "MeanZScore" in lr_df.columns
    assert "FractionSignificant_p05" in lr_df.columns

    # 2. Overview summary
    overview_df = summarize_significance_overview(results)
    assert isinstance(overview_df, pd.DataFrame)
    assert overview_df["TotalEntries"].iloc[0] == 6 * 2 * 2 * len(
        results["filtered_lr_pairs"]
    )
    assert "FractionSignificant_p05" in overview_df.columns
    assert "MedianZScore" in overview_df.columns

    # 3. xarray conversion
    ds = significance_to_xarray(results)
    assert "observed_communication" in ds.data_vars
    assert "pvalue" in ds.data_vars
    assert ds.sizes["condition"] == 6
    assert ds.sizes["sender_rise"] == 2
    assert ds.sizes["receiver_rise"] == 2


def test_runner_cli_parser_and_paths():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "alad",
            "--reference-source",
            "cccrise/data/factors/bal_alad_factors.nc",
            "--n-perms",
            "500",
            "--communication-score",
            "expression_product",
        ]
    )
    assert args.dataset == "alad"
    assert args.reference_source == "cccrise/data/factors/bal_alad_factors.nc"
    assert args.n_perms == 500

    dummy_spec = DatasetSpec(
        key="test_cohort",
        label="Test Cohort",
        loader=lambda **kwargs: anndata.AnnData(),
        condition_column="sample",
        celltype_column="celltype",
        factors_filename="test.nc",
        source_file=Path("test.h5ad"),
    )
    paths = output_paths(dummy_spec, Path("/tmp"))
    assert "lr" in paths
    assert "tensor" not in paths
    assert str(paths["lr"]).endswith("significance_lr_test_cohort.parquet")


def test_patient_block_permutations_keep_patients_together():
    patients = np.array(["a", "b", "c", "c", "d", "e", "e"])
    perms = patient_block_permutations(patients, n_perms=50, random_state=0)
    for perm in perms:
        assert sorted(perm) == list(range(len(patients)))
        # Each multi-sample patient's slots receive one whole block
        for slots in ([2, 3], [5, 6]):
            assert sorted(perm[slots]) in ([2, 3], [5, 6])
    assert len({tuple(p) for p in perms}) > 1


def test_patient_permutation_differential_matches_group_means():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(8, 3))
    case = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    observed, null = patient_permutation_differential(
        scores, case, np.arange(8).astype(str), n_perms=20
    )
    assert np.allclose(observed, scores[case].mean(0) - scores[~case].mean(0))
    assert null.shape == (20, 3)
