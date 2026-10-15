"""
Statistical significance testing and summarization utilities for CCC-RISE.

Implements the within-condition cell-loading permutation test
(`ccc_rise.compute_communication_significance`), with functions to summarize
its per-LR pair and global significance metrics, and the patient-block
permutation of condition-level labels (`patient_permutation_differential`)
that tests group differences in condition-level scores.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import xarray as xr

from ..ccc_rise import (
    compute_communication_significance,
    condition_matrices,
    condition_projections,
)
from ..utils import load_rise_reference


def fit_significance_pipeline(
    adata: anndata.AnnData,
    condition_column: str,
    reference_path: str | Path,
    lr_pairs: pd.DataFrame,
    n_perms: int = 1000,
    communication_score: str = "expression_product",
    complex_sep: str | None = "&",
    complex_agg_method: str = "min",
    hill_k: float = 0.5,
    random_state: int = 42,
    validate: bool = False,
) -> dict[str, Any]:
    """Load a shared RISE fit and compute within-condition communication
    significance.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell count/expression matrix with condition annotations in `.obs`.
    condition_column : str
        Column in `adata.obs` defining patient/sample conditions.
    reference_path : str | Path
        A factors.nc written by `run_ccc_rise` (or `run_alt_comm_score`),
        whose RISE fit is loaded rather than refit.
    lr_pairs : pd.DataFrame
        Ligand-receptor pairs table with columns 'ligand' and 'receptor'.
    n_perms : int, default=1000
        Number of permutations for significance testing.
    communication_score : str, default='expression_product'
        Scoring function ('expression_product', 'expression_mean',
        'expression_gmean', 'expression_hill'), independent of whichever
        score `reference_path` was built with.
    complex_sep : str, default='&'
        Delimiter for multi-subunit protein complexes.
    complex_agg_method : str, default='min'
        Method for aggregating subunits within a complex ('min', 'mean', 'gmean').
    hill_k : float, default=0.5
        Half-saturation constant for Hill communication scoring.
    random_state : int, default=42
        Random seed for permutations.
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
        - 'filtered_lr_pairs': pd.DataFrame of evaluated LR pairs
        - 'condition_names': list[str] of conditions
        - 'r2x_rise': float PARAFAC2 fit quality (of the loaded reference)
    """
    adata = load_rise_reference(reference_path, adata, condition_column)
    r2x_rise = adata.uns["r2x_pf2"]
    condition_names = np.unique(adata.obs[condition_column]).tolist()

    # Within-condition cell-loading permutation test
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
        projections=condition_projections(adata),
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

    return {
        "observed_tensor": observed_tensor,
        "pvalue_tensor": pvalue_tensor,
        "null_mean_tensor": null_mean_tensor,
        "null_std_tensor": null_std_tensor,
        "filtered_lr_pairs": filtered_lr_pairs,
        "condition_names": condition_names,
        "r2x_rise": r2x_rise,
        "n_perms": n_perms,
    }


def summarize_significance_lr(results: dict[str, Any]) -> pd.DataFrame:
    """Summarize per-LR pair communication significance.

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

    return summary.sort_values(by="MeanZScore", ascending=False).reset_index(drop=True)


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

    data_vars: dict[str, Any] = {
        "observed_communication": (dims_4d, results["observed_tensor"]),
        "pvalue": (dims_4d, results["pvalue_tensor"]),
        "null_mean": (dims_4d, results["null_mean_tensor"]),
        "null_std": (dims_4d, results["null_std_tensor"]),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords)


def patient_block_permutations(
    patients: Sequence[str] | np.ndarray, n_perms: int, random_state: int = 0
) -> np.ndarray:
    """`n_perms` reorderings of the conditions (n_perms x n_conditions index
    arrays) that keep each patient's conditions together: single-condition
    patients are shuffled among themselves, and a patient's block of k
    conditions swaps only with other patients' blocks of k. Indexing a
    condition-level label array with a row permutes the labels by patient."""
    rng = np.random.default_rng(random_state)
    patients = np.asarray(patients)
    blocks: dict[int, list[np.ndarray]] = {}
    for patient in pd.unique(patients):
        idx = np.flatnonzero(patients == patient)
        blocks.setdefault(len(idx), []).append(idx)
    perms = np.tile(np.arange(len(patients)), (n_perms, 1))
    for same_size in blocks.values():
        stacked = np.array(same_size)
        for perm in perms:
            perm[stacked.ravel()] = stacked[rng.permutation(len(stacked))].ravel()
    return perms


def patient_permutation_differential(
    scores: np.ndarray,
    case: np.ndarray,
    patients: Sequence[str] | np.ndarray,
    n_perms: int = 10_000,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Case-minus-control mean difference of each column of a conditions x
    features `scores` matrix, observed and under `patient_block_permutations`
    of the boolean `case` labels: (observed (features,), null (n_perms x
    features))."""
    case = np.asarray(case, dtype=bool)

    def weights(c: np.ndarray) -> np.ndarray:
        return c / c.sum() - ~c / (~c).sum()

    perms = patient_block_permutations(patients, n_perms, random_state)
    null = np.stack([weights(case[p]) for p in perms]) @ scores
    return weights(case) @ scores, null
