"""
Statistical significance testing and summarization utilities for CCC-RISE.

Implements the within-condition cell-loading permutation test
(`compute_communication_significance`) and its vectorized communication
scoring, with functions to summarize
its per-LR pair and global significance metrics, and the patient-block
permutation of condition-level labels (`patient_permutation_differential`)
that tests group differences in condition-level scores.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import spmatrix

from ..ccc import combine_communication_score
from ..ccc_rise import (
    calc_communication_score,
    condition_projections,
)
from ..import_data import condition_matrices, import_ligand_receptor_pairs
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


def _resolve_subunit_indices(
    names: pd.Series, gene_index: dict[str, int], complex_sep: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map each ligand/receptor entry (possibly a `complex_sep`-joined multimeric
    complex) to indices into the gene axis, for vectorized expression lookups.

    Returns
    -------
    idx : np.ndarray of shape (n_entries, max_subunits)
        Gene indices for each subunit (padded with 0 where invalid).
    mask : np.ndarray of shape (n_entries, max_subunits)
        True where `idx` holds a real subunit.
    valid : np.ndarray of shape (n_entries,)
        True where every subunit of the entry was found in `gene_index`
        (mirrors `add_complexes_to_expression`, which zeroes out the whole
        complex if any subunit is missing).
    """
    split_names = [
        n.split(complex_sep) if complex_sep is not None else [n] for n in names
    ]
    max_subunits = max(len(s) for s in split_names)
    n = len(split_names)

    idx = np.zeros((n, max_subunits), dtype=np.int64)
    mask = np.zeros((n, max_subunits), dtype=bool)
    valid = np.zeros(n, dtype=bool)

    for i, subunits in enumerate(split_names):
        found = [gene_index.get(s) for s in subunits]
        if all(f is not None for f in found):
            valid[i] = True
            for j, f in enumerate(found):
                idx[i, j] = cast(int, f)
                mask[i, j] = True

    return idx, mask, valid


def _aggregate_subunit_expression(
    matrix: np.ndarray,
    idx: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    agg_method: str,
) -> np.ndarray:
    """
    Reduce subunit expression to a single value per ligand/receptor entry,
    for every rank, reproducing `add_complexes_to_expression` without pandas.

    Parameters
    ----------
    matrix : np.ndarray of shape (rank, genes)
    idx, mask, valid : as returned by `_resolve_subunit_indices`

    Returns
    -------
    np.ndarray of shape (rank, n_entries)
    """
    vals = matrix[:, idx]  # (rank, n_entries, max_subunits)
    counts = np.maximum(mask.sum(axis=1), 1)

    if agg_method == "min":
        out = np.where(mask, vals, np.inf).min(axis=2)
    elif agg_method == "mean":
        out = np.where(mask, vals, 0.0).sum(axis=2) / counts
    elif agg_method == "gmean":
        with np.errstate(divide="ignore"):
            log_vals = np.where(mask, np.log(vals), 0.0)
        out = np.exp(log_vals.sum(axis=2) / counts)
    else:
        raise ValueError(f"{agg_method} is not a valid agg_method")

    out[:, ~valid] = 0.0
    return out


def _fast_interaction_tensor(
    projected_matrices: list[np.ndarray],
    lig_idx: np.ndarray,
    lig_mask: np.ndarray,
    lig_valid: np.ndarray,
    rec_idx: np.ndarray,
    rec_mask: np.ndarray,
    rec_valid: np.ndarray,
    complex_agg_method: str,
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
) -> np.ndarray:
    """Vectorized communication tensor for a fixed LR pair set, for any of
    the scoring strategies in `combine_communication_score` (product, mean,
    gmean, or the CellChat-style Hill function)."""
    # Written into a preallocated output rather than stacked at the end:
    # `np.stack` needs the per-condition slabs and the result live at once, and
    # at cohort scale with a high RISE rank each of those is tens of GB (a
    # 343 x 70 x 70 x 1372 tensor is 18 GB on its own).
    out = None
    for i, matrix in enumerate(projected_matrices):
        lig = _aggregate_subunit_expression(
            matrix, lig_idx, lig_mask, lig_valid, complex_agg_method
        )
        rec = _aggregate_subunit_expression(
            matrix, rec_idx, rec_mask, rec_valid, complex_agg_method
        )
        raw_product = np.einsum("rl,sl->rsl", lig, rec)
        raw_sum = lig[:, np.newaxis, :] + rec[np.newaxis, :, :]
        score = combine_communication_score(
            raw_product,
            raw_sum,
            communication_score=communication_score,
            hill_k=hill_k,
        )
        if out is None:
            out = np.empty((len(projected_matrices), *score.shape), dtype=score.dtype)
        out[i] = score

    if out is None:
        raise ValueError("No projected matrices to build a communication tensor from.")
    return out


def compute_communication_significance(
    X_list: list[np.ndarray],
    projections: list[np.ndarray],
    gene_names: list[str],
    lr_pairs: pd.DataFrame | None = None,
    n_perms: int = 1000,
    random_state: int | None = None,
    complex_sep: str | None = "&",
    complex_agg_method: str = "min",
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    gene_means: np.ndarray | None = None,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Assess the significance of every entry of the cccRISE communication
    tensor against a CellChat-style permutation null.

    CellChat/CellPhoneDB build their null by randomly reassigning cells to
    cluster labels and recomputing per-cluster mean expression. RISE has no
    discrete clusters, only a continuous, per-cell loading onto each
    eigen-state (a row of the PARAFAC2 projection matrix `P_i`). The direct
    generalization is therefore to permute which cell's expression each
    loading row is paired with -- i.e. shuffle the rows of `P_i` per
    condition, holding the raw expression `X_i` and the LR annotation fixed
    -- and recompute the communication tensor. This breaks the real
    cell/eigen-state association while preserving cell counts, the loading
    distribution, and the LR biology, exactly mirroring what a cluster-label
    permutation tests in the discrete case.

    For efficiency, the communication score (any of the strategies in
    `combine_communication_score`, with complexes aggregated via
    `complex_agg_method`) is recomputed for every permutation using a
    vectorized NumPy implementation rather than by rebuilding the tensor
    through `calc_communication_score`/Tensor Cell2Cell each time, which
    dominates runtime for `n_perms` in the hundreds to thousands. The LR
    pair set and gene lookups are resolved once up front; only the cheap
    eigen-state/gene outer products are repeated per permutation.
    `validate=True` checks the vectorized path against the reference
    implementation once, on the observed (unpermuted) tensor.

    Parameters
    ----------
    X_list : list[np.ndarray]
        Per-condition raw expression matrices, shape (cells, genes).
    projections : list[np.ndarray]
        Per-condition RISE/PARAFAC2 projection matrices, shape (cells, rank).
    gene_names : list[str]
        Gene names corresponding to the columns of each `X_list` entry.
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns.
    n_perms : int, default=1000
        Number of eigen-state/cell permutations forming the null distribution.
    random_state : int, optional
        Random seed for reproducibility.
    complex_sep : str, optional, default="&"
        Separator for multimeric complexes.
    complex_agg_method : str, default="min"
        Method to aggregate subunit expression for complexes.
    communication_score : str, default='expression_product'
        Scoring function combining ligand/receptor eigen-state expression
        into a communication score; see
        `cccrise.ccc.combine_communication_score` for the
        available options ('expression_product', 'expression_mean',
        'expression_gmean', 'expression_hill'). Every entry of the null
        distribution and the observed tensor uses the same scoring function,
        so p-values/effect sizes are directly comparable across strategies.
    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`.
    gene_means : np.ndarray, optional
        Per-gene mean expression over the full dataset (`adata.var["means"]`).
        `X_list` here is raw (uncentered) expression, so `P.T @ X` is already
        a projection of non-negative values -- but its entries can still be
        negative, since `P`'s columns are signed contrast axes (see #139).
        When given, `gene_means` is added to each (observed and permuted)
        projected matrix and clipped at 0, exactly as in
        `calc_communication_score`, keeping this null model consistent with
        the sign-safe scoring used by the main pipeline. If None (default),
        matrices are used unchanged, preserving prior behavior.
    validate : bool, default=True
        Cross-check the vectorized observed tensor against
        `calc_communication_score` and raise if they disagree.

    Returns
    -------
    interaction_tensor : np.ndarray
        Observed 4D communication tensor, shape (conditions, rank, rank, n_lr_pairs).
    pvalue_tensor : np.ndarray
        Empirical p-value per entry, same shape as `interaction_tensor`.
    null_mean_tensor : np.ndarray
        Per-entry mean of the null distribution, same shape.
    null_std_tensor : np.ndarray
        Per-entry standard deviation of the null distribution, same shape;
        together with `null_mean_tensor` this gives an effect-size/z-score
        (`(interaction_tensor - null_mean_tensor) / null_std_tensor`) alongside
        the p-value.
    filtered_lr_pairs : pd.DataFrame
        LR pairs corresponding to the tensor's last axis.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    # Only densify legacy scipy sparse matrices; leave other array-likes
    # (e.g. parafac2's mean-centering `SampleArray`) to handle `@` themselves,
    # matching the convention used in `ccc_rise`.
    X_arrays = [
        cast(Any, x).toarray() if isinstance(x, spmatrix) else x for x in X_list
    ]
    dense_P = [np.asarray(p) for p in projections]
    means_row = (
        np.asarray(gene_means).reshape(1, -1) if gene_means is not None else None
    )

    def _project_and_recenter(P: np.ndarray, X: np.ndarray) -> np.ndarray:
        matrix = P.T @ X
        if means_row is not None:
            matrix = np.clip(matrix + means_row, 0.0, None)
        return matrix

    projected_matrices = [
        _project_and_recenter(P, X) for P, X in zip(dense_P, X_arrays)
    ]

    interaction_tensor, filtered_lr_pairs = calc_communication_score(
        projected_matrices=projected_matrices,
        gene_names=gene_names,
        lr_pairs=lr_pairs,
        complex_sep=complex_sep,
        complex_agg_method=complex_agg_method,
        communication_score=communication_score,
        hill_k=hill_k,
    )

    gene_index = {g: i for i, g in enumerate(gene_names)}
    lig_idx, lig_mask, lig_valid = _resolve_subunit_indices(
        filtered_lr_pairs["ligand"], gene_index, complex_sep
    )
    rec_idx, rec_mask, rec_valid = _resolve_subunit_indices(
        filtered_lr_pairs["receptor"], gene_index, complex_sep
    )

    if validate:
        _validate_fast_interaction_tensor(
            interaction_tensor,
            projected_matrices,
            lig_idx,
            lig_mask,
            lig_valid,
            rec_idx,
            rec_mask,
            rec_valid,
            complex_agg_method,
            communication_score,
            hill_k,
        )

    pvalue_tensor, null_mean_tensor, null_std_tensor = _run_permutation_null(
        interaction_tensor,
        dense_P,
        X_arrays,
        _project_and_recenter,
        lig_idx,
        lig_mask,
        lig_valid,
        rec_idx,
        rec_mask,
        rec_valid,
        complex_agg_method,
        communication_score,
        hill_k,
        n_perms,
        random_state,
    )

    return (
        interaction_tensor,
        pvalue_tensor,
        null_mean_tensor,
        null_std_tensor,
        filtered_lr_pairs,
    )


def _validate_fast_interaction_tensor(
    interaction_tensor: np.ndarray,
    projected_matrices: list[np.ndarray],
    lig_idx: np.ndarray,
    lig_mask: np.ndarray,
    lig_valid: np.ndarray,
    rec_idx: np.ndarray,
    rec_mask: np.ndarray,
    rec_valid: np.ndarray,
    complex_agg_method: str,
    communication_score: str,
    hill_k: float,
) -> None:
    """Cross-check the vectorized observed tensor against the reference
    `calc_communication_score` path, raising if they disagree."""
    fast_observed = _fast_interaction_tensor(
        projected_matrices,
        lig_idx,
        lig_mask,
        lig_valid,
        rec_idx,
        rec_mask,
        rec_valid,
        complex_agg_method,
        communication_score=communication_score,
        hill_k=hill_k,
    )
    if not np.allclose(fast_observed, interaction_tensor, rtol=1e-5, atol=1e-8):
        raise RuntimeError(
            "Vectorized communication score does not match "
            "calc_communication_score on the observed tensor."
        )


def _run_permutation_null(
    interaction_tensor: np.ndarray,
    dense_P: list[np.ndarray],
    X_arrays: list[Any],
    project_and_recenter,
    lig_idx: np.ndarray,
    lig_mask: np.ndarray,
    lig_valid: np.ndarray,
    rec_idx: np.ndarray,
    rec_mask: np.ndarray,
    rec_valid: np.ndarray,
    complex_agg_method: str,
    communication_score: str,
    hill_k: float,
    n_perms: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the CellChat-style permutation null and return empirical
    p-values plus the per-entry null mean/std."""
    rng = np.random.default_rng(random_state)
    ge_count = np.zeros_like(interaction_tensor, dtype=np.int64)
    null_sum = np.zeros_like(interaction_tensor, dtype=np.float64)
    null_sumsq = np.zeros_like(interaction_tensor, dtype=np.float64)

    for _ in range(n_perms):
        permuted_matrices = [
            project_and_recenter(P[rng.permutation(P.shape[0]), :], X)
            for P, X in zip(dense_P, X_arrays)
        ]
        null_tensor = _fast_interaction_tensor(
            permuted_matrices,
            lig_idx,
            lig_mask,
            lig_valid,
            rec_idx,
            rec_mask,
            rec_valid,
            complex_agg_method,
            communication_score=communication_score,
            hill_k=hill_k,
        )
        ge_count += null_tensor >= interaction_tensor
        null_sum += null_tensor
        null_sumsq += null_tensor**2

    pvalue_tensor = (ge_count + 1.0) / (n_perms + 1.0)
    null_mean_tensor = null_sum / n_perms
    null_var_tensor = np.maximum(null_sumsq / n_perms - null_mean_tensor**2, 0.0)
    null_std_tensor = np.sqrt(null_var_tensor)

    return pvalue_tensor, null_mean_tensor, null_std_tensor


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
