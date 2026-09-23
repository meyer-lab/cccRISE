"""
Statistical significance testing and summarization utilities for CCC-RISE.

Implements the within-condition cell-loading permutation test
(`ccc_rise.compute_communication_significance`) and the cross-condition
sample-label differential test (`ccc_rise.compute_differential_significance`),
along with functions to summarize per-LR pair, differential, and global
significance metrics.
"""

from typing import Any

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from parafac2.parafac2 import parafac2_nd

from ..ccc_rise import (
    compute_communication_significance,
    compute_differential_significance,
    condition_matrices,
)
from ..import_data import add_cond_idxs


def fit_significance_pipeline(
    adata: anndata.AnnData,
    condition_column: str,
    rise_rank: int,
    lr_pairs: pd.DataFrame,
    n_perms: int = 1000,
    group_column: str | None = None,
    control_label: str | None = None,
    other_label: str | None = None,
    communication_score: str = "expression_product",
    complex_sep: str | None = "&",
    complex_agg_method: str = "min",
    hill_k: float = 0.5,
    max_exact_perms: int = 100_000,
    n_iter_max: int = 100,
    random_state: int = 42,
    validate: bool = False,
    *,
    rise_tol: float,
) -> dict[str, Any]:
    """Fit RISE and compute within-condition and differential communication significance.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell count/expression matrix with condition annotations in `.obs`.
    condition_column : str
        Column in `adata.obs` defining patient/sample conditions.
    rise_rank : int
        Number of RISE eigen-state components.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs table with columns 'ligand' and 'receptor'.
    n_perms : int, default=1000
        Number of permutations for significance testing.
    group_column : str, optional
        Binary group column in `adata.obs` (e.g. disease status). If provided,
        differential significance testing is also performed across groups.
    control_label : str, optional
        Label indicating the reference/control group.
    other_label : str, optional
        Label indicating the comparison/case group.
    communication_score : str, default='expression_product'
        Scoring function ('expression_product', 'expression_mean',
        'expression_gmean', 'expression_hill').
    complex_sep : str, default='&'
        Delimiter for multi-subunit protein complexes.
    complex_agg_method : str, default='min'
        Method for aggregating subunits within a complex ('min', 'mean', 'gmean').
    hill_k : float, default=0.5
        Half-saturation constant for Hill communication scoring.
    max_exact_perms : int, default=100_000
        Maximum combinatorial assignments to enumerate exactly for differential test.
    n_iter_max : int, default=100
        Maximum iterations for PARAFAC2 fitting.
    rise_tol : float
        Convergence tolerance for PARAFAC2 fitting.
    random_state : int, default=42
        Random seed for permutations and PARAFAC2 initialization.
    validate : bool, default=False
        If True, validates vectorized tensor computation against `calc_communication_score`.

    Returns
    -------
    dict[str, Any]
        Significance results containing:
        - 'observed_tensor': 4D ndarray (conditions, senders, receivers, lr_pairs)
        - 'pvalue_tensor': 4D ndarray empirical p-values
        - 'null_mean_tensor': 4D ndarray permutation null mean scores
        - 'null_std_tensor': 4D ndarray permutation null std scores
        - 'diff_observed': 3D ndarray (senders, receivers, lr_pairs) differential score (if group_column given)
        - 'diff_pvalue': 3D ndarray differential p-values (if group_column given)
        - 'diff_null_mean': 3D ndarray differential null means (if group_column given)
        - 'diff_null_std': 3D ndarray differential null stds (if group_column given)
        - 'filtered_lr_pairs': pd.DataFrame of evaluated LR pairs
        - 'condition_names': list[str] of conditions
        - 'r2x_rise': float PARAFAC2 fit quality
    """
    adata = add_cond_idxs(adata, condition_column)
    pf2_output, r2x_rise = parafac2_nd(
        adata,
        rank=rise_rank,
        n_iter_max=n_iter_max,
        tol=rise_tol,
        random_state=random_state,
    )
    dense_P = pf2_output[2]
    condition_names = np.unique(adata.obs[condition_column]).tolist()

    # 2. Within-condition permutation significance test
    X_list = condition_matrices(adata)
    gene_names = list(adata.var_names)
    gene_means = (
        adata.var["means"].to_numpy()
        if "means" in adata.var
        else np.zeros(len(gene_names))
    )

    (
        observed_tensor,
        pvalue_tensor,
        null_mean_tensor,
        null_std_tensor,
        filtered_lr_pairs,
    ) = compute_communication_significance(
        X_list=X_list,
        projections=dense_P,
        gene_names=gene_names,
        lr_pairs=lr_pairs,
        n_perms=n_perms,
        random_state=random_state,
        complex_sep=complex_sep,
        complex_agg_method=complex_agg_method,
        communication_score=communication_score,
        hill_k=hill_k,
        gene_means=gene_means,
        validate=validate,
    )

    # 3. Cross-condition differential significance test (if group info is available)
    diff_observed = None
    diff_pvalue = None
    diff_null_mean = None
    diff_null_std = None

    if group_column is not None and group_column in adata.obs:
        cond_to_group = adata.obs.groupby(condition_column)[group_column].first()
        unique_vals = set(cond_to_group.dropna().unique())
        sample_to_group = None

        if control_label is not None and control_label in unique_vals:
            sample_to_group = np.array(
                [cond_to_group[c] != control_label for c in condition_names],
                dtype=bool,
            )
        elif other_label is not None:
            sample_to_group = np.array(
                [cond_to_group[c] == other_label for c in condition_names],
                dtype=bool,
            )
        else:
            if len(unique_vals) == 2:
                vals_list = list(unique_vals)
                control_label, other_label = str(vals_list[0]), str(vals_list[1])
                sample_to_group = np.array(
                    [cond_to_group[c] == other_label for c in condition_names],
                    dtype=bool,
                )

        if sample_to_group is not None and 0 < sample_to_group.sum() < len(
            sample_to_group
        ):
            (
                diff_observed,
                diff_pvalue,
                diff_null_mean,
                diff_null_std,
            ) = compute_differential_significance(
                interaction_tensor=observed_tensor,
                sample_to_group=sample_to_group,
                n_perms=n_perms,
                max_exact_perms=max_exact_perms,
                random_state=random_state,
            )

    return {
        "observed_tensor": observed_tensor,
        "pvalue_tensor": pvalue_tensor,
        "null_mean_tensor": null_mean_tensor,
        "null_std_tensor": null_std_tensor,
        "diff_observed": diff_observed,
        "diff_pvalue": diff_pvalue,
        "diff_null_mean": diff_null_mean,
        "diff_null_std": diff_null_std,
        "filtered_lr_pairs": filtered_lr_pairs,
        "condition_names": condition_names,
        "r2x_rise": r2x_rise,
        "n_perms": n_perms,
        "group_column": group_column,
        "control_label": control_label,
        "other_label": other_label,
    }


def summarize_significance_lr(results: dict[str, Any]) -> pd.DataFrame:
    """Summarize per-LR pair communication significance and differential testing.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary returned by `fit_significance_pipeline`.

    Returns
    -------
    pd.DataFrame
        Table indexed per LR pair with columns:
        - 'LRPair': LR pair interaction symbol
        - 'Ligand': ligand gene name(s)
        - 'Receptor': receptor gene name(s)
        - 'MeanObserved': mean observed communication score across all conditions & eigen-states
        - 'MeanNull': mean permutation null score
        - 'MeanZScore': empirical effect size (Z-score over null distribution)
        - 'MedianZScore': median Z-score across condition-eigenstate slices
        - 'FractionSignificant_p05': fraction of (condition, sender, receiver) entries with p < 0.05
        - 'FractionSignificant_p01': fraction of entries with p < 0.01
        - 'MaxObserved': maximum observed communication score
        - 'MeanDiffObserved': mean differential communication (if differential test was performed)
        - 'FractionDiffSignificant_p05': fraction of (sender, receiver) differential entries with p < 0.05
    """
    observed = results["observed_tensor"]  # (n_cond, n_send, n_recv, n_lr)
    pvalues = results["pvalue_tensor"]
    null_means = results["null_mean_tensor"]
    null_stds = results["null_std_tensor"]
    lr_df = results["filtered_lr_pairs"].copy()

    # Empirical z-scores per entry: (obs - null_mean) / max(null_std, 1e-3)
    safe_stds = np.maximum(null_stds, 1e-3)
    z_scores = (observed - null_means) / safe_stds

    # Reduction over (conditions, senders, receivers): axes (0, 1, 2)
    mean_obs = np.mean(observed, axis=(0, 1, 2))
    mean_null = np.mean(null_means, axis=(0, 1, 2))
    mean_z = np.mean(z_scores, axis=(0, 1, 2))
    median_z = np.median(z_scores, axis=(0, 1, 2))
    max_obs = np.max(observed, axis=(0, 1, 2))

    frac_sig_05 = np.mean(pvalues < 0.05, axis=(0, 1, 2))
    frac_sig_01 = np.mean(pvalues < 0.01, axis=(0, 1, 2))

    lr_symbols = (
        lr_df["interaction_symbol"].tolist()
        if "interaction_symbol" in lr_df
        else [f"{l}_{r}" for l, r in zip(lr_df["ligand"], lr_df["receptor"])]
    )

    summary = pd.DataFrame(
        {
            "LRPair": lr_symbols,
            "Ligand": lr_df["ligand"].tolist(),
            "Receptor": lr_df["receptor"].tolist(),
            "MeanObserved": mean_obs,
            "MeanNull": mean_null,
            "MeanZScore": mean_z,
            "MedianZScore": median_z,
            "FractionSignificant_p05": frac_sig_05,
            "FractionSignificant_p01": frac_sig_01,
            "MaxObserved": max_obs,
        }
    )

    # Attach differential metrics if available
    if (
        results.get("diff_observed") is not None
        and results.get("diff_pvalue") is not None
    ):
        diff_obs = results["diff_observed"]  # (n_send, n_recv, n_lr)
        diff_pvals = results["diff_pvalue"]

        summary["MeanDiffObserved"] = np.mean(diff_obs, axis=(0, 1))
        summary["FractionDiffSignificant_p05"] = np.mean(diff_pvals < 0.05, axis=(0, 1))
        summary["FractionDiffSignificant_p01"] = np.mean(diff_pvals < 0.01, axis=(0, 1))

    return summary.sort_values(by="MeanZScore", ascending=False).reset_index(drop=True)


def summarize_significance_differential(
    results: dict[str, Any],
    top_n: int | None = None,
) -> pd.DataFrame:
    """Summarize cross-condition differential signaling by sender-receiver-LR triplets.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary returned by `fit_significance_pipeline`.
    top_n : int, optional
        If specified, return only the top N interactions by significance.

    Returns
    -------
    pd.DataFrame
        Table of differential interactions with columns:
        - 'Sender': Sender eigen-state component index (1-based)
        - 'Receiver': Receiver eigen-state component index (1-based)
        - 'LRPair': Ligand-receptor pair symbol
        - 'ObservedDiff': Difference in mean communication (Group 1 - Group 0)
        - 'PValue': Differential permutation p-value
        - 'NullMean': Differential permutation null mean difference
        - 'NullStd': Differential permutation null std
        - 'ZScore': Differential effect size (ObservedDiff - NullMean) / NullStd
    """
    if results.get("diff_observed") is None or results.get("diff_pvalue") is None:
        return pd.DataFrame()

    diff_obs = results["diff_observed"]  # (n_send, n_recv, n_lr)
    diff_pvals = results["diff_pvalue"]
    diff_null_mean = results["diff_null_mean"]
    diff_null_std = results["diff_null_std"]
    lr_df = results["filtered_lr_pairs"]

    lr_symbols = (
        lr_df["interaction_symbol"].tolist()
        if "interaction_symbol" in lr_df
        else [f"{l}_{r}" for l, r in zip(lr_df["ligand"], lr_df["receptor"])]
    )

    n_send, n_recv, n_lr = diff_obs.shape
    rows = []

    for s in range(n_send):
        for r in range(n_recv):
            for l in range(n_lr):
                obs_d = float(diff_obs[s, r, l])
                pval = float(diff_pvals[s, r, l])
                nm = float(diff_null_mean[s, r, l])
                ns = float(diff_null_std[s, r, l])
                z = (obs_d - nm) / max(ns, 1e-3)

                rows.append(
                    {
                        "Sender": s + 1,
                        "Receiver": r + 1,
                        "LRPair": lr_symbols[l],
                        "ObservedDiff": obs_d,
                        "PValue": pval,
                        "NullMean": nm,
                        "NullStd": ns,
                        "ZScore": z,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["PValue", "ZScore"], ascending=[True, False]).reset_index(
        drop=True
    )
    if top_n is not None:
        df = df.head(top_n)
    return df


def summarize_significance_overview(results: dict[str, Any]) -> pd.DataFrame:
    """Summarize global cohort-level significance metrics.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary returned by `fit_significance_pipeline`.

    Returns
    -------
    pd.DataFrame
        Single-row overview DataFrame reporting:
        - 'TotalConditions': number of conditions evaluated
        - 'RiseRank': number of RISE eigen-states
        - 'TotalLRPairs': number of LR pairs evaluated
        - 'TotalEntries': total number of tensor elements tested
        - 'FractionSignificant_p05': proportion of tensor elements with p < 0.05
        - 'FractionSignificant_p01': proportion of tensor elements with p < 0.01
        - 'MedianZScore': median effect size across the full communication tensor
        - 'MeanZScore': mean effect size across the full communication tensor
        - 'R2X_RISE': PARAFAC2 variance explained
    """
    observed = results["observed_tensor"]
    pvalues = results["pvalue_tensor"]
    null_means = results["null_mean_tensor"]
    null_stds = results["null_std_tensor"]

    safe_stds = np.maximum(null_stds, 1e-3)
    z_scores = (observed - null_means) / safe_stds

    n_cond, n_send, _, n_lr = observed.shape
    total_entries = observed.size

    return pd.DataFrame(
        [
            {
                "TotalConditions": n_cond,
                "RiseRank": n_send,
                "TotalLRPairs": n_lr,
                "TotalEntries": total_entries,
                "FractionSignificant_p05": float(np.mean(pvalues < 0.05)),
                "FractionSignificant_p01": float(np.mean(pvalues < 0.01)),
                "MedianZScore": float(np.median(z_scores)),
                "MeanZScore": float(np.mean(z_scores)),
                "R2X_RISE": float(results.get("r2x_rise", np.nan)),
            }
        ]
    )


def significance_to_xarray(results: dict[str, Any]) -> xr.Dataset:
    """Convert permutation significance results to an xarray Dataset for NetCDF storage.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary returned by `fit_significance_pipeline`.

    Returns
    -------
    xr.Dataset
        4D xarray Dataset with coords (condition, sender_rise, receiver_rise, lr_pair)
        and data variables:
        - 'observed_communication'
        - 'pvalue'
        - 'null_mean'
        - 'null_std'
        - 'diff_observed' (if differential test was performed)
        - 'diff_pvalue'
        - 'diff_null_mean'
        - 'diff_null_std'
    """
    condition_names = results["condition_names"]
    rise_rank = results["observed_tensor"].shape[1]
    lr_df = results["filtered_lr_pairs"]
    lr_symbols = (
        lr_df["interaction_symbol"].tolist()
        if "interaction_symbol" in lr_df
        else [f"{l}_{r}" for l, r in zip(lr_df["ligand"], lr_df["receptor"])]
    )

    coords = {
        "condition": condition_names,
        "sender_rise": [f"RISE_{i + 1}" for i in range(rise_rank)],
        "receiver_rise": [f"RISE_{i + 1}" for i in range(rise_rank)],
        "lr_pair": lr_symbols,
    }

    dims_4d = ["condition", "sender_rise", "receiver_rise", "lr_pair"]
    dims_3d = ["sender_rise", "receiver_rise", "lr_pair"]

    data_vars: dict[str, Any] = {
        "observed_communication": (dims_4d, results["observed_tensor"]),
        "pvalue": (dims_4d, results["pvalue_tensor"]),
        "null_mean": (dims_4d, results["null_mean_tensor"]),
        "null_std": (dims_4d, results["null_std_tensor"]),
    }

    if results.get("diff_observed") is not None:
        data_vars["diff_observed"] = (dims_3d, results["diff_observed"])
        data_vars["diff_pvalue"] = (dims_3d, results["diff_pvalue"])
        data_vars["diff_null_mean"] = (dims_3d, results["diff_null_mean"])
        data_vars["diff_null_std"] = (dims_3d, results["diff_null_std"])

    return xr.Dataset(data_vars=data_vars, coords=coords)
