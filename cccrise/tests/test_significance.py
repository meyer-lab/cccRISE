"""
Unit tests for cccrise.validation.significance and cccrise.runners.run_significance.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from cccrise.runners._cli import RISE_TOL
from cccrise.runners.datasets import DatasetSpec
from cccrise.runners.run_significance import build_parser, output_paths
from cccrise.validation.significance import (
    fit_significance_pipeline,
    significance_to_xarray,
    summarize_significance_differential,
    summarize_significance_lr,
    summarize_significance_overview,
)


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
    groups = [
        "Control" if i < n_conditions // 2 else "Disease"
        for i in range(n_conditions)
        for _ in range(cells_per_cond)
    ]
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"sample": samples, "status": groups}),
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


def test_fit_significance_pipeline_end_to_end():
    adata, lr_pairs = _synthetic_cohort_for_significance()
    rise_rank = 3
    n_perms = 20

    results = fit_significance_pipeline(
        adata=adata,
        condition_column="sample",
        rise_rank=rise_rank,
        lr_pairs=lr_pairs,
        n_perms=n_perms,
        group_column="status",
        control_label="Control",
        other_label="Disease",
        random_state=42,
        rise_tol=RISE_TOL,
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

    # Check differential outputs
    assert results["diff_observed"] is not None
    assert results["diff_observed"].shape == (rise_rank, rise_rank, n_lr)
    assert results["diff_pvalue"].shape == (rise_rank, rise_rank, n_lr)
    assert np.all(results["diff_pvalue"] >= 0.0)
    assert np.all(results["diff_pvalue"] <= 1.0)


def test_summaries_and_xarray_conversion():
    adata, lr_pairs = _synthetic_cohort_for_significance()
    results = fit_significance_pipeline(
        adata=adata,
        condition_column="sample",
        rise_rank=2,
        lr_pairs=lr_pairs,
        n_perms=10,
        group_column="status",
        control_label="Control",
        other_label="Disease",
        random_state=42,
        rise_tol=RISE_TOL,
    )

    # 1. LR summary
    lr_df = summarize_significance_lr(results)
    assert isinstance(lr_df, pd.DataFrame)
    assert len(lr_df) == len(results["filtered_lr_pairs"])
    assert "MeanObserved" in lr_df.columns
    assert "MeanZScore" in lr_df.columns
    assert "FractionSignificant_p05" in lr_df.columns
    assert "FractionDiffSignificant_p05" in lr_df.columns

    # 2. Differential summary
    diff_df = summarize_significance_differential(results)
    assert isinstance(diff_df, pd.DataFrame)
    assert len(diff_df) == 2 * 2 * len(results["filtered_lr_pairs"])
    assert "ObservedDiff" in diff_df.columns
    assert "PValue" in diff_df.columns
    assert "ZScore" in diff_df.columns

    # 3. Overview summary
    overview_df = summarize_significance_overview(results)
    assert isinstance(overview_df, pd.DataFrame)
    assert overview_df["TotalEntries"].iloc[0] == 6 * 2 * 2 * len(
        results["filtered_lr_pairs"]
    )
    assert "FractionSignificant_p05" in overview_df.columns
    assert "MedianZScore" in overview_df.columns

    # 4. xarray conversion
    ds = significance_to_xarray(results)
    assert "observed_communication" in ds.data_vars
    assert "pvalue" in ds.data_vars
    assert "diff_observed" in ds.data_vars
    assert ds.sizes["condition"] == 6
    assert ds.sizes["sender_rise"] == 2
    assert ds.sizes["receiver_rise"] == 2


def test_runner_cli_parser_and_paths():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "alad",
            "--rise-rank",
            "25",
            "--n-perms",
            "500",
            "--communication-score",
            "expression_product",
        ]
    )
    assert args.dataset == "alad"
    assert args.rise_rank == 25
    assert args.n_perms == 500

    dummy_spec = DatasetSpec(
        key="test_cohort",
        label="Test Cohort",
        loader=lambda **kwargs: anndata.AnnData(),
        condition_column="sample",
        factors_filename="test.nc",
        source_file=Path("test.h5ad"),
    )
    paths = output_paths(dummy_spec, Path("/tmp"))
    assert "lr" in paths
    assert "differential" in paths
    assert "tensor" not in paths
    assert str(paths["lr"]).endswith("significance_lr_test_cohort.csv")
    assert str(paths["differential"]).endswith(
        "significance_differential_test_cohort.parquet"
    )
