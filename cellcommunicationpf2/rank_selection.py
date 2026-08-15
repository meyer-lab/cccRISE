"""
Rank-selection criteria for the RISE (PARAFAC2) stage of CCC-RISE.

All functions here are computable from the single-cell expression data alone: no
ligand-receptor tensor, no CPD, no downstream classifier or clinical outcome. See
PLAN_RISE.md for the rationale and the exact leakage constraints each function is
designed to respect.

Model recap: X_i ~= P_i diag(a_i) B C^T, where P_i (cells_i x rank) has orthonormal
columns, B is (rank x rank), C is (genes x rank) and shared across conditions, and
a_i is row i of A (n_cond x rank). P_i and a_i are per-condition.
"""

from itertools import pairwise
from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
from parafac2.sample import SampleArray
from parafac2.utils import anndata_to_list, parafac_update, project_data
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array, issparse
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from tensorly.tenalg.core_tenalg.n_mode_product import multi_mode_dot

from .ccc import calculate_r2x
from .import_data import add_cond_idxs

Factors = list[np.ndarray]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def match_components_by_gene_factor(
    C1: np.ndarray, C2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match components of two gene-loading matrices (genes x rank) by |cosine|.

    Uses the Hungarian algorithm on the absolute cosine-similarity matrix, since
    PARAFAC2 components carry an arbitrary sign and ordering ambiguity (never
    compare positionally).

    Returns
    -------
    col_ind : np.ndarray
        For each component of C1 (row order 0..rank1-1), the matched column index
        into C2.
    similarity : np.ndarray
        The |cosine| similarity for each matched pair, same order as col_ind.
    """
    n1 = np.linalg.norm(C1, axis=0, keepdims=True)
    n2 = np.linalg.norm(C2, axis=0, keepdims=True)
    n1 = np.where(n1 > 1e-12, n1, 1.0)
    n2 = np.where(n2 > 1e-12, n2, 1.0)
    cos_sim = np.abs((C1 / n1).T @ (C2 / n2))  # (rank1, rank2)

    row_ind, col_ind = linear_sum_assignment(-cos_sim)
    similarity = cos_sim[row_ind, col_ind]

    # linear_sum_assignment with a square-or-rectangular matrix returns row_ind
    # sorted and of length min(rank1, rank2); re-expand to length rank1 with -1 /
    # NaN for any unmatched components (only occurs if rank1 != rank2).
    rank1 = C1.shape[1]
    full_col_ind = np.full(rank1, -1, dtype=int)
    full_similarity = np.full(rank1, np.nan)
    full_col_ind[row_ind] = col_ind
    full_similarity[row_ind] = similarity
    return full_col_ind, full_similarity


def principal_angle_cosines(C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spaces of C1 and C2.

    Both are (features x rank) matrices, not necessarily orthonormal; they are
    orthonormalized via QR first. Returns the singular values of Q1^T Q2, i.e. the
    cosines of the principal angles between the two subspaces, sorted descending.
    """
    Q1, _ = np.linalg.qr(C1)
    Q2, _ = np.linalg.qr(C2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.clip(s, -1.0, 1.0)


def subspace_similarity(C1: np.ndarray, C2: np.ndarray) -> dict[str, float]:
    """Summarize subspace agreement between two gene-loading matrices via principal angles."""
    cosines = principal_angle_cosines(C1, C2)
    return {
        "mean_cos_principal_angle": float(np.mean(cosines)),
        "min_cos_principal_angle": float(np.min(cosines)),
        "grassmann_distance": float(np.linalg.norm(np.arccos(cosines))),
    }


def component_replication_counts(
    C1: np.ndarray, C2: np.ndarray, thresholds: tuple[float, ...] = (0.6, 0.7, 0.8)
) -> dict[str, float]:
    """Count matched (by |cos| on gene factor C) components exceeding each threshold.

    This is a per-component alternative to averaging FMS over all components: an
    average must decay smoothly as more marginal components are added at higher
    rank, while a count can plateau.
    """
    _, similarity = match_components_by_gene_factor(C1, C2)
    out: dict[str, float] = {"mean_matched_similarity": float(np.nanmean(similarity))}
    for tau in thresholds:
        out[f"n_replicated_tau{tau}"] = float(np.sum(similarity > tau))
    return out


def _subset_genes_with_recomputed_means(
    adata: anndata.AnnData, gene_mask: np.ndarray
) -> anndata.AnnData:
    """Slice an AnnData object to a gene subset and recompute var['means'] for the slice.

    `anndata_to_list` centers each condition's data using `var["means"]`; if genes
    are dropped without recomputing this column, centering becomes inconsistent
    with the sliced data.
    """
    sub = adata[:, gene_mask].copy()
    X = cast(Any, sub.X)
    X = csr_array(X) if issparse(X) else X
    sub.var["means"] = np.asarray(X.mean(axis=0)).ravel()
    return sub


def _fit_rank(
    adata: anndata.AnnData,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> tuple[Factors, list[np.ndarray], float]:
    """Thin wrapper around parafac2_nd returning (factors, projections, R2X)."""
    (_, factors, projections), r2x = parafac2_nd(
        adata, rank=rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
    )
    return factors, projections, r2x


def _projections_to_matrix(
    adata: anndata.AnnData, projections: list[np.ndarray]
) -> np.ndarray:
    """Assemble a per-condition projection list into a single (n_obs, rank) matrix."""
    sg_index = adata.obs["condition_unique_idxs"].to_numpy().astype(int)
    rank = projections[0].shape[1]
    out = np.zeros((adata.n_obs, rank))
    for i, p in enumerate(projections):
        out[sg_index == i, :] = p
    return out


# ---------------------------------------------------------------------------
# Approach 1a - leave-one-sample-out reconstruction (likely-leaky variant)
# ---------------------------------------------------------------------------


def _fit_held_out_condition(
    X_list: list[SampleArray],
    B: np.ndarray,
    C: np.ndarray,
    n_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate (P, a) for a single held-out condition, with B and C frozen.

    Alternates between the projection step (closed-form given the current `a`)
    and the closed-form `a` update used inside the ordinary PARAFAC2 mode-0
    update -- i.e. it runs the *same* per-condition inner loop the training
    algorithm runs, restricted to one condition and with B, C never updated.
    Returns the final projection, `a` row, and reconstruction R2X for this
    condition alone.
    """
    rank = B.shape[0]
    a = np.ones((1, rank))
    norm_x_sq = X_list[0].norm_sq()

    proj = None
    err = norm_x_sq
    for _ in range(n_iter):
        mttkrp, err = project_data(X_list, [a, B, C], norm_x_sq, mode=0)
        factors = parafac_update([a, B, C], mttkrp, mode=0)
        a = factors[0]

    proj_list = project_data(
        X_list, [a, B, C], norm_x_sq, mode=0, return_projections=True
    )
    proj = cast(list[np.ndarray], proj_list)[0]
    _, err = project_data(X_list, [a, B, C], norm_x_sq, mode=0)
    r2x = 1.0 - err / norm_x_sq
    return proj, a[0], float(r2x)


def leave_one_sample_out_r2x(
    adata: anndata.AnnData,
    rank: int,
    condition_column: str,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    n_folds: int | None = None,
) -> pd.DataFrame:
    """Leave-one-sample-out reconstruction R2X at a given rank (Approach 1a).

    For each fold, RISE is fit on all samples but one; B and C from that fit are
    then frozen and used to score the held-out sample after re-estimating only
    its own projection P and condition scaling a (never seen during training).

    Caveat (see PLAN_RISE.md): because P and a are refit from the held-out
    sample's own data, and a larger rank always enlarges the subspace it can
    project into, this quantity may be monotonically increasing in rank just
    like in-sample R2X. That is an empirical question this function is meant to
    answer, not something assumed away here.

    Parameters
    ----------
    n_folds : int | None
        If given, only this many (randomly ordered but deterministic) samples are
        held out, instead of every sample -- a compute-budget knob, since a full
        leave-one-out sweep is O(n_samples) full refits per rank.
    """
    adata = add_cond_idxs(adata, condition_column)
    samples = np.unique(adata.obs[condition_column].to_numpy())
    if n_folds is not None:
        rng = np.random.default_rng(0)
        samples = rng.choice(samples, size=min(n_folds, len(samples)), replace=False)

    rows = []
    for held_out in samples:
        train_mask = (adata.obs[condition_column] != held_out).to_numpy()
        train = add_cond_idxs(adata[train_mask].copy(), condition_column)
        factors, _, train_r2x = _fit_rank(
            train, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
        )
        _, B, C = factors

        test = adata[~train_mask].copy()
        test = add_cond_idxs(test, condition_column)
        test_list = anndata_to_list(test)

        _, _, held_out_r2x = _fit_held_out_condition(test_list, B, C)

        rows.append(
            {
                "Rank": rank,
                "HeldOutSample": held_out,
                "TrainR2X": train_r2x,
                "HeldOutR2X": held_out_r2x,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach 1b - gene x cell block holdout (bi-cross-validation)
# ---------------------------------------------------------------------------


def bicv_reconstruction_error(
    adata: anndata.AnnData,
    rank: int,
    condition_column: str,
    gene_test_frac: float = 0.1,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> dict[str, float]:
    """One fold of the bi-cross-validation scheme (Approach 1b).

    Leakage bookkeeping (every parameter used in the final score is fit on data
    disjoint from the scored block):
      - genes are split into G_tr / G_te
      - cells within each condition are split into N_a / N_b
      - RISE (A, B, C_tr, P) is fit using G_tr genes and ALL cells
      - C_te is estimated by least squares using only G_te genes and N_a cells
      - the score is computed on the G_te x N_b block: P[N_b] and C_te were each
        fit without ever seeing that block.

    Returns a dict with the held-out-block R2X plus the in-sample R2X on the
    training block (G_tr, all cells) as a self-check -- it should match the R2X
    `parafac2_nd` itself reports for the same fit.
    """
    rng = np.random.default_rng(random_state)
    n_genes = adata.n_vars
    n_test_genes = max(1, round(n_genes * gene_test_frac))
    gene_perm = rng.permutation(n_genes)
    te_gene_idx = gene_perm[:n_test_genes]
    tr_gene_idx = gene_perm[n_test_genes:]

    tr_gene_mask = np.zeros(n_genes, dtype=bool)
    tr_gene_mask[tr_gene_idx] = True

    adata = add_cond_idxs(adata, condition_column)
    train_genes = _subset_genes_with_recomputed_means(adata, tr_gene_mask)

    factors, projections, train_r2x_reported = _fit_rank(
        train_genes, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
    )
    A, B, _C_tr = factors
    P = _projections_to_matrix(train_genes, projections)

    sg_index = adata.obs["condition_unique_idxs"].to_numpy(dtype=int)
    n_cells = adata.n_obs
    cell_half = rng.integers(0, 2, size=n_cells).astype(bool)  # True -> N_b

    # Scaled per-cell rank vectors: (P .* A[cond]) @ B, shared by both LS fit and
    # prediction steps.
    a_per_cell = A[sg_index]
    P_scaled = P * a_per_cell
    L = P_scaled @ B  # (n_cells, rank)

    X_full = cast(Any, adata.X)
    X_full = csr_array(X_full) if issparse(X_full) else np.asarray(X_full)
    X_te_genes = cast(Any, X_full[:, te_gene_idx])
    X_te_genes = (
        X_te_genes.toarray() if issparse(X_te_genes) else np.asarray(X_te_genes)
    )

    na_mask = ~cell_half
    nb_mask = cell_half

    mean_te_on_na = X_te_genes[na_mask].mean(axis=0)
    Y_tr = X_te_genes[na_mask] - mean_te_on_na
    L_tr = L[na_mask]

    C_te, *_ = np.linalg.lstsq(L_tr, Y_tr, rcond=None)  # (rank, n_test_genes)

    Y_test = X_te_genes[nb_mask] - mean_te_on_na
    L_test = L[nb_mask]
    pred = L_test @ C_te

    sse = float(np.sum((Y_test - pred) ** 2))
    sst = float(np.sum(Y_test**2))
    held_out_r2x = 1.0 - sse / sst if sst > 0 else np.nan

    # Self-check: in-sample R2X on the training block should reproduce parafac2's
    # own reported number for this fit (same genes, same cells).
    train_block_r2x = train_r2x_reported

    return {
        "Rank": rank,
        "HeldOutBlockR2X": held_out_r2x,
        "TrainBlockR2X": train_block_r2x,
        "NTestGenes": n_test_genes,
        "NTrainGenes": int(tr_gene_mask.sum()),
    }


# ---------------------------------------------------------------------------
# Approach 3 - permutation null for R2X
# ---------------------------------------------------------------------------


def permute_within_sample_genes(
    adata: anndata.AnnData, condition_column: str, random_state: int | None = None
) -> anndata.AnnData:
    """Independently permute each gene's values across cells, within each sample.

    Preserves each gene's marginal distribution exactly (same multiset of values
    per gene per sample) while destroying cross-gene structure -- the "no
    structure at all" null. Do NOT permute the whole matrix at once, which would
    also destroy the per-cell depth/library-size structure.
    """
    rng = np.random.default_rng(random_state)
    out = adata.copy()
    was_sparse = issparse(out.X)
    X = cast(Any, out.X)
    X = csr_array(X).toarray() if was_sparse else np.asarray(X).copy()

    cond = out.obs[condition_column].to_numpy()
    for sample in np.unique(cond):
        block_idx = np.where(cond == sample)[0]
        block = X[block_idx]
        rand = rng.random(block.shape)
        order = np.argsort(rand, axis=0)
        X[block_idx] = np.take_along_axis(block, order, axis=0)

    out.X = csr_array(X) if was_sparse else X
    out = add_cond_idxs(out, condition_column)
    out.var["means"] = X.mean(axis=0)
    return out


def permute_condition_labels(
    adata: anndata.AnnData, condition_column: str, random_state: int | None = None
) -> anndata.AnnData:
    """Randomly reassign cells to samples (weaker null: cell content is untouched).

    Brackets "no sample-specific structure" against the stronger gene-permutation
    null's "no structure at all".
    """
    rng = np.random.default_rng(random_state)
    out = adata.copy()
    shuffled = rng.permutation(out.obs[condition_column].to_numpy())
    out.obs[condition_column] = shuffled
    out = add_cond_idxs(out, condition_column)
    return out


def permutation_null_r2x_sweep(
    adata: anndata.AnnData,
    ranks: list[int],
    condition_column: str,
    n_null_replicates: int = 2,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Observed vs. null R2X across ranks (Approach 3), for both null types."""
    adata = add_cond_idxs(adata, condition_column)
    rows = []
    for rank in ranks:
        _, _, obs_r2x = _fit_rank(
            adata, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
        )
        rows.append(
            {"Rank": rank, "NullType": "observed", "Replicate": 0, "R2X": obs_r2x}
        )

        for rep in range(n_null_replicates):
            seed = None if random_state is None else random_state * 1000 + rep
            gene_null = permute_within_sample_genes(
                adata, condition_column, random_state=seed
            )
            _, _, gene_null_r2x = _fit_rank(
                gene_null,
                rank,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=random_state,
            )
            rows.append(
                {
                    "Rank": rank,
                    "NullType": "gene_permutation",
                    "Replicate": rep,
                    "R2X": gene_null_r2x,
                }
            )

            label_null = permute_condition_labels(
                adata, condition_column, random_state=seed
            )
            _, _, label_null_r2x = _fit_rank(
                label_null,
                rank,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=random_state,
            )
            rows.append(
                {
                    "Rank": rank,
                    "NullType": "condition_label_permutation",
                    "Replicate": rep,
                    "R2X": label_null_r2x,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach 4 - component replication via split-half / bootstrap
# ---------------------------------------------------------------------------


def split_half_replication(
    adata: anndata.AnnData,
    ranks: list[int],
    condition_column: str,
    n_splits: int = 3,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    thresholds: tuple[float, ...] = (0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """Split *samples* (not cells) into two halves and fit each independently.

    Tests whether a program replicates across people/patients, matching on the
    gene factor C (the interpretable, cross-fit-comparable mode).
    """
    adata = add_cond_idxs(adata, condition_column)
    samples = np.unique(adata.obs[condition_column].to_numpy())
    rng = np.random.default_rng(random_state)

    rows = []
    for split in range(n_splits):
        perm = rng.permutation(samples)
        half = len(perm) // 2
        half_a, half_b = perm[:half], perm[half:]

        mask_a = adata.obs[condition_column].isin(half_a).to_numpy()
        mask_b = adata.obs[condition_column].isin(half_b).to_numpy()
        X_a = add_cond_idxs(adata[mask_a].copy(), condition_column)
        X_b = add_cond_idxs(adata[mask_b].copy(), condition_column)

        for rank in ranks:
            factors_a, _, r2x_a = _fit_rank(
                X_a, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
            )
            factors_b, _, r2x_b = _fit_rank(
                X_b, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
            )
            counts = component_replication_counts(
                factors_a[2], factors_b[2], thresholds
            )
            subspace = subspace_similarity(factors_a[2], factors_b[2])
            rows.append(
                {
                    "Rank": rank,
                    "Split": split,
                    "R2X_A": r2x_a,
                    "R2X_B": r2x_b,
                    **counts,
                    **subspace,
                }
            )

    return pd.DataFrame(rows)


def bootstrap_replication(
    adata: anndata.AnnData,
    ranks: list[int],
    condition_column: str,
    n_boot: int = 3,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    thresholds: tuple[float, ...] = (0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """Per-component replication under the existing stratified cell bootstrap.

    Companion to `split_half_replication`: this tests robustness to sequencing
    depth (resampling cells within sample), not replication across patients.
    """
    from .utils import resample

    adata = add_cond_idxs(adata, condition_column)
    rows = []
    for rank in ranks:
        factors_full, _, r2x_full = _fit_rank(
            adata, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
        )
        for b in range(n_boot):
            boot = resample(adata, condition_name=condition_column, random_seed=b)
            boot = add_cond_idxs(boot, condition_column)
            factors_boot, _, r2x_boot = _fit_rank(
                boot, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
            )
            counts = component_replication_counts(
                factors_full[2], factors_boot[2], thresholds
            )
            subspace = subspace_similarity(factors_full[2], factors_boot[2])
            rows.append(
                {
                    "Rank": rank,
                    "Boot": b,
                    "R2X_Full": r2x_full,
                    "R2X_Boot": r2x_boot,
                    **counts,
                    **subspace,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach 5 - nested subspace stability across consecutive ranks
# ---------------------------------------------------------------------------


def nested_rank_subspace_stability(
    adata: anndata.AnnData,
    ranks: list[int],
    condition_column: str,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> pd.DataFrame:
    """How much of the rank-r gene-factor subspace is contained in rank-(r+1)'s.

    Sharp drops indicate the model reorganizing rather than refining as rank
    increases; `ranks` must be given in ascending, consecutive-enough order for
    this comparison to be meaningful (the function only compares adjacent pairs
    in the list, whatever their gap).
    """
    adata = add_cond_idxs(adata, condition_column)
    fits: dict[int, np.ndarray] = {}
    r2x_by_rank: dict[int, float] = {}
    for rank in ranks:
        factors, _, r2x = _fit_rank(
            adata, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state
        )
        fits[rank] = factors[2]
        r2x_by_rank[rank] = r2x

    rows = []
    for r_lo, r_hi in pairwise(ranks):
        subspace = subspace_similarity(fits[r_lo], fits[r_hi])
        rows.append(
            {
                "RankLo": r_lo,
                "RankHi": r_hi,
                "R2X_Lo": r2x_by_rank[r_lo],
                "R2X_Hi": r2x_by_rank[r_hi],
                **subspace,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach 6 - degeneracy / conditioning diagnostics
# ---------------------------------------------------------------------------


def degeneracy_diagnostics(
    adata: anndata.AnnData,
    ranks: list[int],
    condition_column: str,
    n_inits: int = 3,
    n_iter_max: int = 100,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Cheap, resampling-free diagnostics collected from single fits at each rank.

    Also runs `n_inits` random initializations per rank to report init-dependence
    of both the fitted subspace (via principal angles) and the final R2X --
    init-dependence appearing at rank r is a hard usability ceiling.
    """
    adata = add_cond_idxs(adata, condition_column)
    rows = []
    for rank in ranks:
        init_factors = []
        init_r2x = []
        for seed in range(n_inits):
            factors, _, r2x = _fit_rank(
                adata, rank, n_iter_max=n_iter_max, tol=tol, random_state=seed
            )
            init_factors.append(factors)
            init_r2x.append(r2x)

        C0 = init_factors[0][2]
        B0 = init_factors[0][1]

        n1 = np.linalg.norm(C0, axis=0, keepdims=True)
        n1 = np.where(n1 > 1e-12, n1, 1.0)
        C0_norm = C0 / n1
        cos_mat = np.abs(C0_norm.T @ C0_norm)
        np.fill_diagonal(cos_mat, 0.0)
        max_pairwise_cos = float(np.max(cos_mat)) if rank > 1 else float("nan")

        cond_B = float(np.linalg.cond(B0))
        cond_C = float(np.linalg.cond(C0))

        weight_norms = np.linalg.norm(C0, axis=0) * np.linalg.norm(B0, axis=0)
        weight_norms_sorted = np.sort(weight_norms)[::-1]
        weight_spread = (
            float(weight_norms_sorted[0] / weight_norms_sorted[-1])
            if weight_norms_sorted[-1] > 1e-12
            else float("inf")
        )

        init_cosines = []
        for i in range(1, n_inits):
            sim = subspace_similarity(C0, init_factors[i][2])
            init_cosines.append(sim["mean_cos_principal_angle"])

        rows.append(
            {
                "Rank": rank,
                "R2X_mean": float(np.mean(init_r2x)),
                "R2X_std": float(np.std(init_r2x)),
                "MaxPairwiseAbsCos_C": max_pairwise_cos,
                "CondNumber_B": cond_B,
                "CondNumber_C": cond_C,
                "WeightSpread": weight_spread,
                "InitSubspaceAgreement_mean": (
                    float(np.mean(init_cosines)) if init_cosines else float("nan")
                ),
                "InitSubspaceAgreement_min": (
                    float(np.min(init_cosines)) if init_cosines else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CPD rank-selection criteria for the interaction tensor (PLAN_INTERACTION.md)
# =============================================================================
#
# The interaction tensor is (condition x sender-state x receiver-state x
# LR-pair): small, dense, and fully in memory, so unlike RISE there is no
# per-row projection that must be refit from a held-out cell -- masking
# entries directly is a safe, non-leaking holdout here (contrast with
# PLAN_RISE.md's approach-1a caveat). CP components carry the same sign/order
# ambiguity as PARAFAC2; matching always uses `match_components_by_gene_factor`
# / `component_replication_counts` on |cos|, applied to the LR-pair factor
# `D` (index 3), which is the interpretable, cross-fit-comparable mode for
# this tensor (the analogue of RISE's gene factor `C`).


# ---------------------------------------------------------------------------
# Approach A - missing-entry cross-validation (primary recommendation)
# ---------------------------------------------------------------------------


def cp_masked_reconstruction_error(
    tensor: np.ndarray,
    rank: int,
    mask_frac: float = 0.1,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> dict[str, float]:
    """One fold of entry-masked CP cross-validation.

    A random `mask_frac` of tensor entries are held out (mask=0), CP is fit
    with `tensorly`'s built-in EM-style masked ALS, and R2X is scored on the
    held-out entries only using the same normalization as `calculate_r2x`
    (sum-of-squares ratio) so held-out and in-sample numbers are comparable.
    """
    rng = np.random.default_rng(random_state)
    observed = rng.random(tensor.shape) >= mask_frac  # True/1 = observed
    mask = observed.astype(float)

    cp_weights, cp_factors = parafac(
        tensor=tensor,
        rank=rank,
        mask=mask,
        n_iter_max=n_iter_max,
        init=init,
        normalize_factors=True,
        linesearch=True,
        random_state=random_state,
    )
    recon = cp_to_tensor((cp_weights, cp_factors))

    held = ~observed
    sse = float(np.sum((tensor[held] - recon[held]) ** 2))
    sst = float(np.sum(tensor[held] ** 2))
    held_out_r2x = 1.0 - sse / sst if sst > 0 else np.nan
    in_sample_r2x = calculate_r2x(cp_weights, cp_factors, tensor)

    return {
        "Rank": rank,
        "HeldOutR2X": held_out_r2x,
        "InSampleR2X": in_sample_r2x,
        "MaskFrac": mask_frac,
        "NHeldOut": int(held.sum()),
    }


def cp_masked_reconstruction_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    n_masks: int = 5,
    mask_frac: float = 0.1,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Approach A across a rank grid, averaged over `n_masks` independent masks."""
    rows = []
    for rank in ranks:
        for m in range(n_masks):
            seed = None if random_state is None else random_state * 1000 + m
            res = cp_masked_reconstruction_error(
                tensor,
                rank,
                mask_frac=mask_frac,
                n_iter_max=n_iter_max,
                init=init,
                random_state=seed,
            )
            res["Mask"] = m
            rows.append(res)
    return pd.DataFrame(rows)


def cp_condition_holdout_error(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
    conditions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Structured holdout variant of Approach A: hold out whole conditions.

    For each held-out condition, CP is fit with that condition's entire slab
    masked out (shared sender/receiver/LR factors B, C, D never see it), then
    only that condition's own loading is re-estimated by closed-form least
    squares against the frozen factors, and reconstruction is scored on the
    real (unmasked) held-out slab. This speaks directly to generalization
    across patients, unlike entry-masking.
    """
    from tensorly.tenalg.core_tenalg._khatri_rao import khatri_rao

    n_cond = tensor.shape[0]
    which = conditions if conditions is not None else np.arange(n_cond)

    rows = []
    for held in which:
        mask = np.ones(tensor.shape, dtype=float)
        mask[held] = 0.0
        cp_weights, cp_factors = parafac(
            tensor=tensor,
            rank=rank,
            mask=mask,
            n_iter_max=n_iter_max,
            init=init,
            normalize_factors=True,
            linesearch=True,
            random_state=random_state,
        )
        B, C, D = cp_factors[1], cp_factors[2], cp_factors[3]
        design = khatri_rao([B, C, D]) * cp_weights  # fold weights in once
        y = tensor[held].reshape(-1)
        a_held, *_ = np.linalg.lstsq(design, y, rcond=None)
        recon_held = (design @ a_held).reshape(tensor.shape[1:])

        sse = float(np.sum((tensor[held] - recon_held) ** 2))
        sst = float(np.sum(tensor[held] ** 2))
        r2x = 1.0 - sse / sst if sst > 0 else np.nan
        rows.append(
            {"Rank": rank, "HeldOutCondition": int(held), "HeldOutConditionR2X": r2x}
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach B - permutation null / parallel analysis, and per-component
# significance (doubles as Reviewer #1 comment 6's requested null model)
# ---------------------------------------------------------------------------


def permute_lr_within_fiber(
    tensor: np.ndarray, random_state: int | None = None
) -> np.ndarray:
    """Independently permute the LR-pair mode within each (cond, sender, recv)
    fiber. Preserves each fiber's marginal, destroys which LR pair carries
    which score -- the weaker of the two nulls implemented here."""
    rng = np.random.default_rng(random_state)
    shp = tensor.shape
    flat = tensor.reshape(-1, shp[-1]).copy()
    for i in range(flat.shape[0]):
        flat[i] = flat[i, rng.permutation(shp[-1])]
    return flat.reshape(shp)


def permute_eigenstate_labels(
    tensor: np.ndarray, random_state: int | None = None
) -> np.ndarray:
    """Independently permute sender and receiver eigen-state labels, per
    condition. Stronger null than the LR-fiber permutation: it also destroys
    which sender/receiver pair talks to which."""
    rng = np.random.default_rng(random_state)
    n_cond = tensor.shape[0]
    out = np.empty_like(tensor)
    for i in range(n_cond):
        send_perm = rng.permutation(tensor.shape[1])
        recv_perm = rng.permutation(tensor.shape[2])
        out[i] = tensor[i][send_perm][:, recv_perm]
    return out


def cpd_permutation_null_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    n_null_replicates: int = 2,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Observed vs. null R2X across CPD ranks (Approach B), for both null types."""
    rows = []
    for rank in ranks:
        w, f = parafac(
            tensor=tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            normalize_factors=True,
            linesearch=True,
            random_state=random_state,
        )
        rows.append(
            {
                "Rank": rank,
                "NullType": "observed",
                "Replicate": 0,
                "R2X": calculate_r2x(w, f, tensor),
            }
        )

        for rep in range(n_null_replicates):
            seed = None if random_state is None else random_state * 1000 + rep

            lr_null = permute_lr_within_fiber(tensor, random_state=seed)
            w2, f2 = parafac(
                tensor=lr_null,
                rank=rank,
                n_iter_max=n_iter_max,
                init=init,
                normalize_factors=True,
                linesearch=True,
                random_state=random_state,
            )
            rows.append(
                {
                    "Rank": rank,
                    "NullType": "lr_within_fiber",
                    "Replicate": rep,
                    "R2X": calculate_r2x(w2, f2, lr_null),
                }
            )

            es_null = permute_eigenstate_labels(tensor, random_state=seed)
            w3, f3 = parafac(
                tensor=es_null,
                rank=rank,
                n_iter_max=n_iter_max,
                init=init,
                normalize_factors=True,
                linesearch=True,
                random_state=random_state,
            )
            rows.append(
                {
                    "Rank": rank,
                    "NullType": "eigenstate_labels",
                    "Replicate": rep,
                    "R2X": calculate_r2x(w3, f3, es_null),
                }
            )

    return pd.DataFrame(rows)


def cpd_component_significance(
    tensor: np.ndarray,
    rank: int,
    n_null: int = 20,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Sketch of a per-component empirical significance test (Reviewer #1
    major comment 6): reuses the Approach-B permutation machinery.

    For each observed component, its explained-variance fraction
    (weight^2 / total sum-of-squares, valid since `normalize_factors=True`
    gives unit-norm factor columns) is compared to the distribution of the
    *best* component's explained-variance fraction across `n_null`
    LR-within-fiber null fits at the same rank. Using the null's best
    component (rather than a positionally-matched one, which is meaningless
    under permutation) as the comparison baseline is what makes this
    calibrated: it asks "could a component this strong have arisen with no
    real communication structure at all," for every observed component
    independently.
    """
    w, _f = parafac(
        tensor=tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        init=init,
        normalize_factors=True,
        linesearch=True,
        random_state=random_state,
    )
    total_var = float(np.sum(tensor**2))
    obs_ev = (w**2) / total_var if total_var > 0 else np.full(rank, np.nan)

    null_best_ev = np.empty(n_null)
    for i in range(n_null):
        seed = None if random_state is None else random_state * 1000 + i
        null_tensor = permute_lr_within_fiber(tensor, random_state=seed)
        wn, _fn = parafac(
            tensor=null_tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            normalize_factors=True,
            linesearch=True,
            random_state=random_state,
        )
        null_total_var = float(np.sum(null_tensor**2))
        null_ev = (
            (wn**2) / null_total_var if null_total_var > 0 else np.full(rank, np.nan)
        )
        null_best_ev[i] = float(np.max(null_ev))

    pvals = [float(np.mean(null_best_ev >= ev)) for ev in obs_ev]
    return pd.DataFrame(
        {
            "Rank": rank,
            "Component": np.arange(1, rank + 1),
            "ExplainedVarFrac": obs_ev,
            "NullBestEV_mean": float(np.mean(null_best_ev)),
            "EmpiricalP": pvals,
        }
    )


# ---------------------------------------------------------------------------
# Approach C - core consistency diagnostic (CORCONDIA)
# ---------------------------------------------------------------------------


def corcondia(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> float:
    """Core consistency diagnostic (Bro & Kiers 2003).

    Fits CP at `rank`, projects the tensor onto the pseudo-inverted factor
    matrices to obtain the implied Tucker core, and scores how close that
    core is to the superdiagonal identity: 100 while the model is genuinely
    multilinear, falling off a cliff once over-factored. Known to be
    pessimistic under high noise -- use as corroboration, not a sole
    criterion (see PLAN_INTERACTION.md Approach C).
    """
    w, f = parafac(
        tensor=tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        init=init,
        normalize_factors=True,
        linesearch=True,
        random_state=random_state,
    )
    factors = [f[0] * w] + list(f[1:])  # fold weights into mode 0
    pinvs = [np.linalg.pinv(mat) for mat in factors]
    core = multi_mode_dot(tensor, pinvs, modes=list(range(len(factors))))

    ideal = np.zeros((rank,) * len(factors))
    idx = np.arange(rank)
    ideal[(idx,) * len(factors)] = 1.0
    return float(100.0 * (1.0 - np.sum((core - ideal) ** 2) / rank))


def corcondia_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    rows = [
        {
            "Rank": rank,
            "CORCONDIA": corcondia(
                tensor,
                rank,
                n_iter_max=n_iter_max,
                init=init,
                random_state=random_state,
            ),
        }
        for rank in ranks
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach D - replicable component counts, multiple resampling axes
# ---------------------------------------------------------------------------


def resample_tensor_conditions(
    tensor: np.ndarray, random_state: int | None = None
) -> np.ndarray:
    """Bootstrap over conditions/patients (mode 0), with replacement.

    Companion to `tensor.resample_tensor` (which bootstraps the LR-pair mode
    and tests robustness to the LR database): this tests whether a program
    replicates across people, which is the claim the biology rests on.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, tensor.shape[0], size=tensor.shape[0])
    return tensor[idx]


def split_half_conditions(
    tensor: np.ndarray, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split conditions/patients into two independent halves."""
    rng = np.random.default_rng(random_state)
    n_cond = tensor.shape[0]
    perm = rng.permutation(n_cond)
    half = n_cond // 2
    return tensor[perm[:half]], tensor[perm[half:]]


def cpd_replication_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    resample_axis: str = "lr",
    n_boot: int = 5,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
    thresholds: tuple[float, ...] = (0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """Per-component replicable-count sweep (Approach D), matched on the
    LR-pair factor D. `resample_axis` selects what is resampled:
      - "lr": bootstrap the LR-pair mode (robustness to the LR database).
      - "conditions": bootstrap conditions/patients with replacement.
      - "splithalf_conditions": split conditions into two independent halves.
    Different axes answer different questions and are reported separately,
    per the plan's explicit instruction not to conflate them.
    """
    from .tensor import resample_tensor

    rng = np.random.default_rng(random_state)
    rows = []
    for rank in ranks:
        _w_full, f_full = parafac(
            tensor=tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            normalize_factors=True,
            linesearch=True,
            random_state=random_state,
        )
        for b in range(n_boot):
            seed = int(rng.integers(0, 2**31 - 1))

            if resample_axis == "lr":
                t_boot = resample_tensor(tensor)
                _w_boot, f_boot = parafac(
                    tensor=t_boot,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=random_state,
                )
                counts = component_replication_counts(f_full[3], f_boot[3], thresholds)
            elif resample_axis == "conditions":
                t_boot = resample_tensor_conditions(tensor, random_state=seed)
                _w_boot, f_boot = parafac(
                    tensor=t_boot,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=random_state,
                )
                counts = component_replication_counts(f_full[3], f_boot[3], thresholds)
            elif resample_axis == "splithalf_conditions":
                t_a, t_b = split_half_conditions(tensor, random_state=seed)
                _w_a, f_a = parafac(
                    tensor=t_a,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=random_state,
                )
                _w_b, f_b = parafac(
                    tensor=t_b,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=random_state,
                )
                counts = component_replication_counts(f_a[3], f_b[3], thresholds)
            else:
                raise ValueError(f"Unknown resample_axis: {resample_axis!r}")

            rows.append(
                {"Rank": rank, "ResampleAxis": resample_axis, "Boot": b, **counts}
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Approach E - degeneracy and multi-start diagnostics
# ---------------------------------------------------------------------------


def cpd_degeneracy_diagnostics(
    tensor: np.ndarray,
    ranks: list[int],
    n_inits: int = 3,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Multi-start degeneracy/conditioning diagnostics for the CP fit.

    Reports the classic two-factor-degeneracy signature (signed, not
    absolute, pairwise triple congruence across B, C, D approaching -1),
    condition numbers per factor, weight spread, and init-to-init agreement
    of both R2X and the D-factor subspace. `init="random"` by default to
    match `figureS1c_d.py`'s COVID setting -- pass `init="svd"` to check
    whether ALAD's deterministic init masks the same sensitivity.
    """
    rows = []
    for rank in ranks:
        init_factors = []
        init_r2x = []
        for seed in range(n_inits):
            rs = seed if random_state is None else random_state + seed
            w, f = parafac(
                tensor=tensor,
                rank=rank,
                n_iter_max=n_iter_max,
                init=init,
                normalize_factors=True,
                linesearch=True,
                random_state=rs,
            )
            init_factors.append((w, f))
            init_r2x.append(calculate_r2x(w, f, tensor))

        w0, f0 = init_factors[0]
        cong = np.ones((rank, rank))
        for mat in (f0[1], f0[2], f0[3]):
            n = np.linalg.norm(mat, axis=0, keepdims=True)
            n = np.where(n > 1e-12, n, 1.0)
            cos = (mat / n).T @ (mat / n)
            cong *= cos
        np.fill_diagonal(cong, 0.0)
        max_pairwise_cong = float(np.max(cong)) if rank > 1 else float("nan")
        min_pairwise_cong = float(np.min(cong)) if rank > 1 else float("nan")

        cond_numbers = {
            f"CondNumber_{name}": float(np.linalg.cond(mat))
            for name, mat in zip("ABCD", f0, strict=True)
        }

        weight_norms = np.sort(np.abs(w0))[::-1]
        weight_spread = (
            float(weight_norms[0] / weight_norms[-1])
            if weight_norms[-1] > 1e-12
            else float("inf")
        )

        init_subspace = [
            subspace_similarity(f0[3], init_factors[i][1][3])[
                "mean_cos_principal_angle"
            ]
            for i in range(1, n_inits)
        ]

        rows.append(
            {
                "Rank": rank,
                "R2X_mean": float(np.mean(init_r2x)),
                "R2X_std": float(np.std(init_r2x)),
                "MaxPairwiseTripleCongruence": max_pairwise_cong,
                "MinPairwiseTripleCongruence": min_pairwise_cong,
                "WeightSpread": weight_spread,
                "InitSubspaceAgreement_mean": (
                    float(np.mean(init_subspace)) if init_subspace else float("nan")
                ),
                "InitSubspaceAgreement_min": (
                    float(np.min(init_subspace)) if init_subspace else float("nan")
                ),
                **cond_numbers,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 4 - joint (RISE x CPD) rank selection
# ---------------------------------------------------------------------------


def rise_cpd_joint_heatmap(
    adata_raw: anndata.AnnData,
    lr_pairs: pd.DataFrame,
    rise_ranks: list[int],
    cpd_ranks: list[int],
    condition_column: str,
    mask_frac: float = 0.1,
    n_masks: int = 2,
    rise_n_iter_max: int = 100,
    rise_tol: float = 1e-6,
    cp_n_iter_max: int = 500,
    cp_init: str = "random",
    random_state: int = 0,
) -> pd.DataFrame:
    """2-D grid of held-out imputation error (Approach A) over (RISE rank x
    CPD rank). For each RISE rank, RISE is fit once and the interaction
    tensor built from it; CPD ranks are then swept with masked
    cross-validation on that one tensor, so the expensive step (RISE
    refitting) is not repeated per CPD rank. Budget this last, at coarse
    resolution -- it is the most expensive item in the plan.
    """
    from .tensor import calculate_interaction_tensor

    rows = []
    for rise_rank in rise_ranks:
        ad = add_cond_idxs(adata_raw.copy(), condition_column)
        (_, _factors, projections), rise_r2x = parafac2_nd(
            ad,
            rank=rise_rank,
            n_iter_max=rise_n_iter_max,
            tol=rise_tol,
            random_state=random_state,
        )
        sg_index = ad.obs["condition_unique_idxs"].to_numpy(dtype=int)
        ad.obsm["projections"] = np.zeros((ad.n_obs, rise_rank))
        for i, p in enumerate(projections):
            ad.obsm["projections"][sg_index == i, :] = p

        tensor = calculate_interaction_tensor(ad, lr_pairs)

        for cpd_rank in cpd_ranks:
            for m in range(n_masks):
                seed = random_state * 1000 + m
                res = cp_masked_reconstruction_error(
                    tensor,
                    cpd_rank,
                    mask_frac=mask_frac,
                    n_iter_max=cp_n_iter_max,
                    init=cp_init,
                    random_state=seed,
                )
                rows.append(
                    {
                        "RISE_Rank": rise_rank,
                        "RISE_R2X": rise_r2x,
                        "CPD_Rank": cpd_rank,
                        "Mask": m,
                        **res,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 5 - consequence for the ALAD predictive analysis
# ---------------------------------------------------------------------------


def cpd_rank_auc_sweep(
    tensor: np.ndarray,
    sample_to_group_bin: np.ndarray,
    ranks: list[int],
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int = 0,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Cross-validated AUC of the condition factor A at each CPD rank.

    For *reporting only*, alongside the unsupervised criterion (Approach A),
    to quantify how much AUC changes at an unsupervised-selected rank versus
    the predictive-performance-optimized published rank (PLAN_INTERACTION.md
    Sec 5). The unsupervised criterion must be computed, and its selected
    rank recorded, before this function's results are ever inspected --
    calling it does not itself violate that ordering, but consuming its
    output to choose a rank would.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    y = np.asarray(sample_to_group_bin)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    rows = []
    for rank in ranks:
        w, f = parafac(
            tensor=tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            normalize_factors=True,
            linesearch=True,
            random_state=random_state,
        )
        A = f[0] * w  # fold weights into the condition factor
        clf = LogisticRegression(
            solver="saga", l1_ratio=1.0, max_iter=10000, random_state=random_state
        )
        scores = cross_val_score(clf, A, y, scoring="roc_auc", cv=cv)
        rows.append(
            {
                "Rank": rank,
                "AUC_mean": float(np.mean(scores)),
                "AUC_std": float(np.std(scores)),
            }
        )

    return pd.DataFrame(rows)
