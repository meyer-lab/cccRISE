from collections.abc import Sequence
from itertools import combinations
from math import comb
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
from scipy.sparse import spmatrix
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.decomposition import parafac

from .ccc import (
    build_context_ccc_tensor,
    calculate_r2x,
    combine_communication_score,
    get_genes_from_complexes,
)
from .import_data import condition_matrices, import_ligand_receptor_pairs


def calc_communication_score(
    projected_matrices: list[np.ndarray],
    gene_names: list[str] | None = None,
    lr_pairs: pd.DataFrame | None = None,
    complex_sep: str | None = None,
    complex_agg_method: str = "min",
    gene_means: np.ndarray | None = None,
    communication_score: str = "expression_product",
    hill_k: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate cell-cell communication scores using build_context_ccc_tensor
    from Tensor Cell2Cell for all conditions at once.

    Parameters:
    -----------
    projected_matrices : list[np.ndarray]
        List of matrices of shape (rank, genes) representing projected cell expressions
        across different conditions
    gene_names : list[str], optional
        List of gene names corresponding to columns in the matrices
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns
    gene_means : np.ndarray, optional
        Per-gene mean expression over the full dataset, i.e. the same vector
        that was subtracted from every condition slice before fitting
        RISE/PARAFAC2 (`adata.var["means"]`). `projected_matrices` holds
        eigen-state loadings relative to that mean, so entries can be
        negative even though real expression cannot. When `gene_means` is
        given, it is added back to each matrix before scoring, turning
        `prot_a_exp`/`prot_b_exp` into an approximation of actual
        non-negative expression rather than a mean-centered loading: this is
        the fix for the communication-score sign ambiguity in issue #139 (a
        product of two centered loadings collapses to only the interaction
        term of `(mean + deviation) x (mean + deviation)`, which lets two
        negative deviations produce a spurious positive score with no
        relationship to actual signaling strength). Small negative residuals
        left by the linear reconstruction are clipped to 0. If None
        (default), matrices are used unchanged, preserving prior behavior.
    communication_score : str, default='expression_product'
        Scoring function combining each ligand/receptor pair's eigen-state
        expression into a communication score; see
        `cccrise.ccc.combine_communication_score` for the
        available options ('expression_product', 'expression_mean',
        'expression_gmean', 'expression_hill').
    hill_k : float, default=0.5
        Half-saturation constant, used only when
        `communication_score='expression_hill'`.

    Returns:
    --------
    np.ndarray
        4D interaction tensor of shape (conditions, rank, rank, n_lr_pairs)
    pd.DataFrame
        The filtered ligand-receptor pairs that correspond to the tensor's last dimension.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(projected_matrices[0].shape[1])]

    if gene_means is not None:
        means_row = np.asarray(gene_means).reshape(1, -1)
        projected_matrices = [
            np.clip(matrix + means_row, 0.0, None) for matrix in projected_matrices
        ]

    # Convert matrices to DataFrames (genes as rows, cells as columns)
    rnaseq_matrices = []
    for matrix in projected_matrices:
        df = pd.DataFrame(
            matrix.T,
            index=pd.Index(gene_names),
            columns=pd.Index([f"rank_{j}" for j in range(matrix.shape[0])]),
        )
        rnaseq_matrices.append(df)

    # Rename columns to match Cell2Cell convention
    if "ligand" in lr_pairs.columns and "receptor" in lr_pairs.columns:
        lr_pairs_renamed = lr_pairs.rename(columns={"ligand": "A", "receptor": "B"})
    else:
        lr_pairs_renamed = lr_pairs

    # Handle complexes if requested
    if complex_sep is not None:
        if verbose:
            print("Getting expression values for protein complexes")
        _, _, _, _, complexes = get_genes_from_complexes(
            ppi_data=lr_pairs_renamed,
            complex_sep=complex_sep,
            interaction_columns=("A", "B"),
        )
        mod_rnaseq_matrices = [
            add_complexes_to_expression(
                rnaseq, complexes, agg_method=complex_agg_method
            )
            for rnaseq in rnaseq_matrices
        ]
    else:
        mod_rnaseq_matrices = [df.copy() for df in rnaseq_matrices]

    # Generate communication tensor for all contexts
    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
        rnaseq_matrices=mod_rnaseq_matrices,
        ppi_data=lr_pairs_renamed,
        how="inner",
        communication_score=communication_score,
        hill_k=hill_k,
        complex_sep=complex_sep,
        upper_letter_comparison=False,
        interaction_columns=("A", "B"),
        group_ppi_by=None,
        group_ppi_method="gmean",
        verbose=verbose,
    )

    # Filter the original lr_pairs to match the pairs used in the tensor
    lr_pair_names = (
        lr_pairs["ligand"].astype(str) + "^" + lr_pairs["receptor"].astype(str)
    )
    filtered_lr_pairs = lr_pairs[lr_pair_names.isin(ppi_names)].reset_index(drop=True)

    # Convert to numpy and transpose to expected format
    # From: (context, ppi_idx, rank, rank)
    # To:   (context, rank, rank, ppi_idx)
    interaction_tensor = np.array(tensors)
    interaction_tensor = np.transpose(interaction_tensor, (0, 2, 3, 1))

    return interaction_tensor, filtered_lr_pairs


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
    tensors = []
    for matrix in projected_matrices:
        lig = _aggregate_subunit_expression(
            matrix, lig_idx, lig_mask, lig_valid, complex_agg_method
        )
        rec = _aggregate_subunit_expression(
            matrix, rec_idx, rec_mask, rec_valid, complex_agg_method
        )
        raw_product = np.einsum("rl,sl->rsl", lig, rec)
        raw_sum = lig[:, np.newaxis, :] + rec[np.newaxis, :, :]
        tensors.append(
            combine_communication_score(
                raw_product,
                raw_sum,
                communication_score=communication_score,
                hill_k=hill_k,
            )
        )

    return np.stack(tensors, axis=0)


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
    # matching the convention used elsewhere in this module (see `ccc_rise`).
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

    rng = np.random.default_rng(random_state)
    ge_count = np.zeros_like(interaction_tensor, dtype=np.int64)
    null_sum = np.zeros_like(interaction_tensor, dtype=np.float64)
    null_sumsq = np.zeros_like(interaction_tensor, dtype=np.float64)

    for _ in range(n_perms):
        permuted_matrices = [
            _project_and_recenter(P[rng.permutation(P.shape[0]), :], X)
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

    return (
        interaction_tensor,
        pvalue_tensor,
        null_mean_tensor,
        null_std_tensor,
        filtered_lr_pairs,
    )


def compute_differential_significance(
    interaction_tensor: np.ndarray,
    sample_to_group: Sequence[bool] | np.ndarray,
    n_perms: int = 1000,
    random_state: int | None = None,
    max_exact_perms: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Test whether communication tensor entries differ between two sample
    groups (e.g. disease severity, ALAD status), against a group-label
    permutation null.

    This is the cross-condition counterpart to
    `compute_communication_significance`, which only tests whether an entry
    is unusual relative to random eigen-state/cell pairing *within* a single
    condition. It has no notion of "differs between groups of samples" --
    exactly the gap identified when comparing it to scDiffCom's own
    permutation scheme (`scDiffCom:::run_stat_iteration`/`run_stat_analysis`,
    R), which runs two distinct tests: a within-condition test (cell-type-label
    permutation) and a separate cross-condition differential-expression (DE)
    test (condition-label permutation, resampled at the *sample* level when
    replicate samples exist within a condition, so within-sample correlation
    is preserved rather than broken by shuffling individual cells).

    Adaptation to cccRISE: cccRISE's tensor already has one slice per
    sample along its first axis (`interaction_tensor.shape[0]`, the same
    axis as the CP condition factor `A`) -- unlike scDiffCom, which starts
    from raw per-cell data and must resample cells/samples to build each
    condition's expression before scoring. This means the sample-level
    permutation scDiffCom performs to preserve within-sample structure is
    already the *only* kind of permutation possible here: this function
    permutes `sample_to_group` -- which existing, already-computed
    per-sample tensor slice counts as "group 0" vs "group 1" -- rather than
    resampling cells or rebuilding the tensor from expression. That keeps
    every sample's internal (eigen-state, LR-pair) structure completely
    intact under the null, exactly matching the property scDiffCom's
    sample-level resampling is designed to preserve, at a fraction of the
    cost (no projections, expression, or communication scores are
    recomputed -- only a group-mean difference over precomputed slices).

    Because the number of conditions/samples is typically small (tens, not
    thousands), this enumerates the null **exactly** -- every way to split
    the samples into two groups of the observed sizes -- whenever that count
    is at or below `max_exact_perms`, rather than Monte Carlo sampling it.
    This directly answers Reviewer #1's small-sample concern (comment #5)
    for this particular test: with n=12 samples split e.g. 6-6, there are
    only `math.comb(12, 6) == 924` possible splits, so the exact null is
    cheap and the reported p-value is exact, not an approximation with its
    own Monte Carlo error. Above `max_exact_perms`, it falls back to
    `n_perms` random label permutations (matching
    `compute_communication_significance`'s convention).

    Parameters
    ----------
    interaction_tensor : np.ndarray
        Observed 4D communication tensor, shape
        (conditions/samples, rank, rank, n_lr_pairs) -- the same tensor
        `compute_communication_significance`/`calc_communication_score`
        return.
    sample_to_group : Sequence[bool] or np.ndarray
        Boolean (or 0/1) group label per condition/sample, same order and
        length as `interaction_tensor`'s first axis and as the CP condition
        factor `A` (see `logreg.ccc_rise_logreg_weights`, which uses the same
        `sample_to_group` convention). Must contain both groups.
    n_perms : int, default=1000
        Number of random group-label permutations, used only when the exact
        null (see above) would be larger than `max_exact_perms`.
    random_state : int, optional
        Random seed for the Monte Carlo fallback; unused when the null is
        enumerated exactly.
    max_exact_perms : int, default=100_000
        Enumerate the null exactly when `comb(n_conditions, n_group1)` is at
        or below this; otherwise fall back to `n_perms` random permutations.

    Returns
    -------
    observed_diff : np.ndarray
        Group-mean difference (group 1 minus group 0) for every tensor
        entry, shape (rank, rank, n_lr_pairs).
    pvalue : np.ndarray
        Two-sided empirical p-value per entry (`|null diff| >= |observed
        diff|`), same shape -- two-sided because, unlike
        `compute_communication_significance`'s one-sided "is this bigger
        than random" test, "different between groups" has no preferred
        direction, matching scDiffCom's own DE test convention.
    null_mean : np.ndarray
        Per-entry mean of the null group-difference distribution.
    null_std : np.ndarray
        Per-entry standard deviation of the null group-difference
        distribution.
    """
    interaction_tensor = np.asarray(interaction_tensor)
    group = np.asarray(sample_to_group).astype(bool)
    n_conditions = interaction_tensor.shape[0]
    if group.shape != (n_conditions,):
        raise ValueError(
            f"sample_to_group must have one entry per condition/sample "
            f"({n_conditions}), got shape {group.shape}"
        )
    n_group1 = int(group.sum())
    n_group0 = n_conditions - n_group1
    if n_group1 == 0 or n_group0 == 0:
        raise ValueError("sample_to_group must contain both groups")

    def _group_diff(mask: np.ndarray) -> np.ndarray:
        return interaction_tensor[mask].mean(axis=0) - interaction_tensor[~mask].mean(
            axis=0
        )

    observed_diff = _group_diff(group)
    ge_count = np.zeros_like(observed_diff, dtype=np.int64)
    null_sum = np.zeros_like(observed_diff, dtype=np.float64)
    null_sumsq = np.zeros_like(observed_diff, dtype=np.float64)

    total_splits = comb(n_conditions, n_group1)
    if total_splits <= max_exact_perms:
        n_used = 0
        for group1_idx in combinations(range(n_conditions), n_group1):
            mask = np.zeros(n_conditions, dtype=bool)
            mask[list(group1_idx)] = True
            null_diff = _group_diff(mask)
            ge_count += np.abs(null_diff) >= np.abs(observed_diff)
            null_sum += null_diff
            null_sumsq += null_diff**2
            n_used += 1
        pvalue = ge_count / n_used
    else:
        rng = np.random.default_rng(random_state)
        for _ in range(n_perms):
            null_diff = _group_diff(rng.permutation(group))
            ge_count += np.abs(null_diff) >= np.abs(observed_diff)
            null_sum += null_diff
            null_sumsq += null_diff**2
        n_used = n_perms
        pvalue = (ge_count + 1.0) / (n_used + 1.0)

    null_mean = null_sum / n_used
    null_var = np.maximum(null_sumsq / n_used - null_mean**2, 0.0)
    null_std = np.sqrt(null_var)

    return observed_diff, pvalue, null_mean, null_std


def validate_cp_rank(cp_rank: int, tensor_shape: tuple[int, ...]) -> None:
    """Reject a CP rank larger than the tensor's smallest mode.

    Above that, tensorly's SVD initialization falls back to a full SVD of a
    mode unfolding -- an allocation quadratic in the unfolding's long axis, so
    hundreds of GiB for a realistic communication tensor -- and pads the excess
    components with uniform random columns, starting them from noise rather
    than from the data.

    The communication tensor's modes are ``(n_conditions, rise_rank,
    rise_rank, n_lr_pairs)``, so in practice the constraint is
    ``min(n_conditions, rise_rank)``.

    Raises
    ------
    ValueError
        If `cp_rank` exceeds the smallest mode of `tensor_shape`.
    """
    smallest_mode = min(tensor_shape)
    if cp_rank > smallest_mode:
        raise ValueError(
            f"cp_rank={cp_rank} exceeds the smallest mode of the communication "
            f"tensor (shape {tuple(tensor_shape)}, smallest mode {smallest_mode}). "
            f"Use cp_rank <= {smallest_mode}, or raise the binding mode -- for "
            "the communication tensor that means more conditions or a larger "
            "rise_rank."
        )


def ccc_rise(
    adata: anndata.AnnData,
    rise_rank: int,
    n_iter_max: int,
    tol: float,
    cp_rank: int | None = None,
    random_state: int | None = None,
    complex_sep: str | None = None,
    lr_pairs: pd.DataFrame | None = None,
    svd_init: str = "svd",
) -> tuple[tuple, float, pd.DataFrame]:
    """
    Perform RISE on an AnnData object, followed by
    CP decomposition on the resulting interaction tensor.

    Parameters:
    -----------
    adata : anndata.AnnData
        Annotated data object with expression data in `.X`
    rise_rank : int
        Rank for the PARAFAC2 decomposition (RISE)
    n_iter_max : int
        Maximum number of iterations for PARAFAC2
    tol : float
        Convergence tolerance for PARAFAC2
    cp_rank : int, optional
        Rank for the CP decomposition. If None, defaults to `rise_rank`.
    random_state : int, optional
        Seed for reproducibility

    Returns:
    --------
    tuple
        A tuple containing:
        - A nested tuple with CP results and projections: ((cp_weights, cp_factors), projections)
        - The R2X (variance explained) of the CP decomposition on the interaction tensor.
        - The filtered ligand-receptor pairs DataFrame used in the analysis.
    """
    gene_names = list(adata.var_names)
    X_list = condition_matrices(adata)

    # PARAFAC2 decomposition
    pf2_output, r2x_pf2 = parafac2_nd(
        adata, rank=rise_rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
    )
    pf2_weights, pf2_factors, projections = pf2_output

    adata.uns["Pf2_weights"] = pf2_weights
    adata.uns["Pf2_A"] = pf2_factors[0]
    adata.uns["Pf2_B"] = pf2_factors[1]
    adata.uns["Pf2_C"] = pf2_factors[2]
    adata.uns["r2x_pf2"] = r2x_pf2

    # Project matrices
    projected_matrices = [
        proj.T @ tensor for proj, tensor in zip(projections, X_list, strict=True)
    ]

    # Per-gene mean subtracted before fitting RISE (see `prepare_dataset`); added
    # back during scoring to keep communication scores non-negative (issue #139).
    gene_means = (
        adata.var["means"].to_numpy()
        if "means" in adata.var
        else np.zeros(len(gene_names))
    )

    # Calculate cell-cell communication scores
    interaction_tensors, filtered_lr_pairs = calc_communication_score(
        projected_matrices,
        gene_names=gene_names,
        complex_sep=complex_sep,
        lr_pairs=lr_pairs,
        gene_means=gene_means,
    )

    cp_rank = cp_rank if cp_rank is not None else rise_rank

    # Print shape of interaction tensors
    print(f"Interaction tensors shape: {interaction_tensors.shape}")

    validate_cp_rank(cp_rank, interaction_tensors.shape)

    # CP decomposition with explicit random initialization
    cp_weights, cp_factors = parafac(
        interaction_tensors,
        cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init=svd_init,  # Use random initialization
        normalize_factors=True,
        random_state=random_state,
    )

    # Calculate R2X for the CP decomposition of the interaction tensor
    final_R2X = calculate_r2x(cp_weights, cp_factors, interaction_tensors)

    return ((cp_weights, cp_factors), projections), final_R2X, filtered_lr_pairs


def standardize_cc_pf2(
    weights: np.ndarray, factors: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Standardize CP factors for better interpretability.
    This function expects all inputs to be NumPy arrays on the CPU.

    Parameters
    ----------
    factors : list[np.ndarray]
        CP factors from the decomposition.
    weights : np.ndarray, optional
        Component weights from the CP decomposition. If None, they are initialized to ones.

    Returns
    -------
    tuple
        (weights, factors) after standardization.
    """
    # Order components by condition variance
    gini = np.var(factors[0], axis=0) / np.mean(factors[0], axis=0)
    gini_idx = np.argsort(gini)
    factors = [f[:, gini_idx] for f in factors]
    if weights is not None:
        weights = weights[gini_idx]

    weights, factors = cp_flip_sign(cp_normalize((weights, factors)), mode=1)

    return weights, factors


def calc_communication_score_pseudobulk(
    pseudobulk_matrices_df: list[pd.DataFrame],
    gene_names: list[str] | None = None,
    lr_pairs: pd.DataFrame | None = None,
    complex_sep: str | None = "&",
    complex_agg_method: str = "min",
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate cell-cell communication scores for pseudobulk
    Parameters:
    -----------
    pseudobulk_matrices_df : list[pd.DataFrame]
        List of dataframes of shape (genes, groupby)
    gene_names : list[str], optional
        List of gene names corresponding to columns in the matrices
    lr_pairs : pd.DataFrame, optional
        DataFrame with 'ligand' and 'receptor' columns
    complex_sep : str, default="&"
        Symbol that separates the protein subunits in a multimeric complex
    complex_agg_method : str, default="min"
        Method to aggregate expression values for complexes
    verbose : bool, default=False
        Print verbose output
    Returns:
    --------
    np.ndarray
        4D interaction tensor of shape (conditions, sender, receiver, n_lr_pairs)
    pd.DataFrame
        The filtered ligand-receptor pairs that correspond to the tensor's last dimension.
    """
    if lr_pairs is None:
        lr_pairs = import_ligand_receptor_pairs()

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(pseudobulk_matrices_df[0].shape[1])]

    # Rename columns to match Cell2Cell convention
    if "ligand" in lr_pairs.columns and "receptor" in lr_pairs.columns:
        lr_pairs_renamed = lr_pairs.rename(columns={"ligand": "A", "receptor": "B"})
        interaction_columns = ("A", "B")
    else:
        lr_pairs_renamed = lr_pairs
        interaction_columns = ("A", "B")

    # Generate expression values for protein complexes in PPI data
    if complex_sep is not None:
        if verbose:
            print("Getting expression values for protein complexes")
        _, _, _, _, complexes = get_genes_from_complexes(
            ppi_data=lr_pairs_renamed,
            complex_sep=complex_sep,
            interaction_columns=interaction_columns,
        )
        mod_rnaseq_matrices = [
            add_complexes_to_expression(
                rnaseq, complexes, agg_method=complex_agg_method
            )
            for rnaseq in pseudobulk_matrices_df
        ]
    else:
        mod_rnaseq_matrices = [df.copy() for df in pseudobulk_matrices_df]

    tensors, _, _, ppi_names, _ = build_context_ccc_tensor(
        rnaseq_matrices=mod_rnaseq_matrices,
        ppi_data=lr_pairs_renamed,
        how="inner",
        communication_score="expression_product",
        complex_sep=complex_sep,
        upper_letter_comparison=True,
        interaction_columns=interaction_columns,
        group_ppi_by=None,
        group_ppi_method="gmean",
        verbose=verbose,
    )

    # Only keep ppi_names to match the pairs used in the tensor in the interaction symbol column of the lr_pairs DataFrame
    lr_pair_names = (
        lr_pairs["ligand"].astype(str) + "^" + lr_pairs["receptor"].astype(str)
    )
    filtered_lr_pairs = lr_pairs[lr_pair_names.isin(ppi_names)].reset_index(drop=True)

    # Convert to numpy and transpose to expected format
    # From: (context, ppi_idx, sender, receiver)
    # To:   (context, sender, receiver, ppi_idx)
    interaction_tensor = np.array(tensors)
    interaction_tensor = np.transpose(interaction_tensor, (0, 2, 3, 1))

    return interaction_tensor, filtered_lr_pairs


def pseudobulk_cp_decomposition(
    interaction_tensors: np.ndarray,
    cp_rank: int,
    n_iter_max: int,
    tol: float | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """
    Perform non-negative CP decomposition on the pseudobulk interaction tensors.
    Parameters
    ----------
    interaction_tensors : np.ndarray
        The interaction tensors to decompose.
    cp_rank : int
        The rank for the CP decomposition.
    n_iter_max : int
        The maximum number of iterations for the decomposition.
    tol: float | None
        Tolerance for convergence.
    random_state : int | None
        Random seed for reproducibility.
    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], float]
        A tuple containing the CP weights, CP factors, and the R2X value.
    """
    # CP decomposition
    cp_weights, cp_factors = parafac(
        interaction_tensors,
        cp_rank,
        n_iter_max=n_iter_max,
        tol=tol,
        init="svd",  # Use SVD initialization
        normalize_factors=True,
        random_state=random_state,
    )

    # Calculate R2X for the CP decomposition of the interaction tensor
    r2x = calculate_r2x(cp_weights, cp_factors, interaction_tensors)

    return cp_weights, cp_factors, r2x


def save_ccc_rise_results(
    X: anndata.AnnData,
    cpd_factors: list[np.ndarray],
    weights: np.ndarray,
    lr_pairs: pd.DataFrame,
):
    """Save CPD results in an AnnData object."""
    X.uns["A"] = cpd_factors[0]  # Condition factor
    X.uns["B"] = cpd_factors[1]  # Sender cell types factor
    X.uns["C"] = cpd_factors[2]  # Receiver cell types factor
    X.uns["D"] = cpd_factors[3]  # LR pairs factor
    X.uns["lr_pairs"] = lr_pairs["interaction_symbol"].values  # LR pairs
    X.uns["weights"] = weights  # Component weights

    return X


def add_complexes_to_expression(rnaseq_data, complexes, agg_method="min"):
    """
    Adds multimeric complexes into the gene expression matrix.
    Their gene expressions are the minimum expression value
    among the respective subunits composing them.

    Parameters
    ----------
    rnaseq_data : pandas.DataFrame
        Gene expression data for RNA-seq experiment. Columns are
        cell-types/tissues/samples and rows are genes.

    complexes : dict
        Dictionary where keys are the complex names in the list of PPIs, while
        values are list of subunits for the respective complex names.

    agg_method : str, default='min'
        Method to aggregate the expression value of multiple genes in a
        complex.

        - 'min' : Minimum expression value among all genes.
        - 'mean' : Average expression value among all genes.
        - 'gmean' : Geometric mean expression value among all genes.

    Returns
    -------
    tmp_rna : pandas.DataFrame
        Gene expression data for RNA-seq experiment containing multimeric
        complex names. Their gene expressions are the minimum expression value
        among the respective subunits composing them. Columns are
        cell-types/tissues/samples and rows are genes.
    """
    tmp_rna = rnaseq_data.copy()
    for k, v in complexes.items():
        if isinstance(v, set):
            v = list(v)
        elif isinstance(v, list):
            pass  # No need to convert, already a list
        else:
            raise TypeError(
                "Values in the `complexes`dictionary must be sets or lists."
            )
        if all(g in tmp_rna.index for g in v):
            df = tmp_rna.loc[v, :]
            if agg_method == "min":
                tmp_rna.loc[k] = df.min().values.tolist()
            elif agg_method == "mean":
                tmp_rna.loc[k] = df.mean().values.tolist()
            elif agg_method == "gmean":
                tmp_rna.loc[k] = df.apply(
                    lambda x: np.exp(np.mean(np.log(x)))
                ).values.tolist()
            else:
                raise ValueError(f"{agg_method} is not a valid agg_method")
        else:
            tmp_rna.loc[k] = [0] * tmp_rna.shape[1]
    return tmp_rna
