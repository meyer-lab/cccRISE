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
