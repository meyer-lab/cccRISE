"""
Rank-selection criteria for RISE (PARAFAC2) and interaction-tensor (CPD) stages.
"""

from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array, issparse
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from .import_data import add_cond_idxs

Factors = list[np.ndarray]


# =============================================================================
# Helper utilities
# =============================================================================


def calculate_r2x(weights: np.ndarray, factors: Factors, tensor: np.ndarray) -> float:
    """Calculate reconstruction R2X for a CP decomposition."""
    reconstruction = cp_to_tensor((weights, factors))
    norm_tensor_sq = np.sum(tensor**2)
    if norm_tensor_sq == 0:
        return 0.0
    norm_residual_sq = np.sum((tensor - reconstruction) ** 2)
    return float(1.0 - norm_residual_sq / norm_tensor_sq)


def match_components_by_gene_factor(
    C1: np.ndarray, C2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily (via Hungarian algorithm) match components between two fits.

    Matches based on the maximum absolute cosine similarity of their factor
    matrices. Resolves arbitrary component permutation and column signs.

    Parameters
    ----------
    C1 : np.ndarray
        (features, rank1) factor matrix.
    C2 : np.ndarray
        (features, rank2) factor matrix.

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
    """Summarize subspace agreement between two factor matrices via principal angles."""
    cosines = principal_angle_cosines(C1, C2)
    return {
        "mean_cos_principal_angle": float(np.mean(cosines)),
        "min_cos_principal_angle": float(np.min(cosines)),
        "grassmann_distance": float(np.linalg.norm(np.arccos(cosines))),
    }


def component_replication_counts(
    C1: np.ndarray, C2: np.ndarray, thresholds: tuple[float, ...] = (0.6, 0.7, 0.8)
) -> dict[str, float]:
    """Count matched (by |cos| on factor matrix) components exceeding each threshold."""
    _, similarity = match_components_by_gene_factor(C1, C2)
    out: dict[str, float] = {"mean_matched_similarity": float(np.nanmean(similarity))}
    for tau in thresholds:
        out[f"n_replicated_tau{tau}"] = float(np.sum(similarity > tau))
    return out


# =============================================================================
# RISE (PARAFAC2) rank-selection criteria
# =============================================================================


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


def bicv_reconstruction_error(
    adata: anndata.AnnData,
    rank: int,
    condition_column: str,
    gene_test_frac: float = 0.1,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> dict[str, float]:
    """Calculate one fold of the bi-cross-validation (BiCV) reconstruction error.

    Leakage bookkeeping (every parameter used in the final score is fit on data
    disjoint from the scored block):
      - genes are split into G_tr / G_te
      - cells within each condition are split into N_a / N_b
      - RISE (A, B, C_tr, P) is fit using G_tr genes and ALL cells
      - C_te is estimated by least squares using only G_te genes and N_a cells
      - the score is computed on the G_te x N_b block: P[N_b] and C_te were each
        fit without ever seeing that block.

    Returns a dict with the held-out-block R2X plus the in-sample R2X on the
    training block (G_tr, all cells) as a reference check.
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

    return {
        "Rank": rank,
        "HeldOutBlockR2X": held_out_r2x,
        "TrainBlockR2X": train_r2x_reported,
        "NTestGenes": n_test_genes,
        "NTrainGenes": int(tr_gene_mask.sum()),
    }


def run_bicv(
    adata: anndata.AnnData,
    ranks: list[int] | range,
    condition_column: str,
    n_folds: int = 3,
    gene_test_frac: float = 0.1,
    n_iter_max: int = 100,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Run bi-cross-validation rank selection across specified ranks and folds.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression data.
    ranks : list[int] | range
        Collection of ranks to evaluate.
    condition_column : str
        Column name in `adata.obs` specifying sample or condition.
    n_folds : int, default=3
        Number of cross-validation folds per rank.
    gene_test_frac : float, default=0.1
        Fraction of genes held out for testing in each fold.
    n_iter_max : int, default=100
        Maximum number of iterations for PARAFAC2 fitting.
    tol : float, default=1e-6
        Tolerance threshold for PARAFAC2 fitting convergence.

    Returns
    -------
    pd.DataFrame
        DataFrame containing results for each rank and fold with columns:
        'Rank', 'Fold', 'HeldOutBlockR2X', 'TrainBlockR2X', 'NTestGenes', 'NTrainGenes'.
    """
    rows = []
    for rank in ranks:
        for fold in range(n_folds):
            res = bicv_reconstruction_error(
                adata=adata,
                rank=rank,
                condition_column=condition_column,
                gene_test_frac=gene_test_frac,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=fold,
            )
            res["Fold"] = fold
            rows.append(res)
    return pd.DataFrame(rows)


# =============================================================================
# CPD rank-selection criteria for the interaction tensor
# =============================================================================


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
    (denominator is the sum of squares of the held-out raw values).
    """
    rng = np.random.default_rng(random_state)
    mask = rng.uniform(size=tensor.shape) >= mask_frac
    if not np.any(~mask):
        mask[tuple(rng.integers(0, s) for s in tensor.shape)] = False

    w, f = parafac(
        tensor=tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        init=init,
        mask=mask.astype(float),
        normalize_factors=True,
        linesearch=True,
        random_state=random_state,
    )

    pred = cp_to_tensor((w, f))
    held_out_mask = ~mask
    actual = tensor[held_out_mask]
    predicted = pred[held_out_mask]

    sse = float(np.sum((actual - predicted) ** 2))
    sst = float(np.sum(actual**2))
    held_out_r2x = 1.0 - sse / sst if sst > 0 else np.nan

    in_sample_r2x = calculate_r2x(w, f, tensor)

    return {
        "Rank": rank,
        "HeldOutR2X": held_out_r2x,
        "InSampleR2X": in_sample_r2x,
        "NHeldOut": int(np.sum(held_out_mask)),
        "NTrain": int(np.sum(mask)),
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
    """Entry-masked cross-validation swept across ranks and mask replicates."""
    rows = []
    for rank in ranks:
        for m in range(n_masks):
            seed = None if random_state is None else random_state * 1000 + m
            res = cp_masked_reconstruction_error(
                tensor,
                rank=rank,
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
    conditions: np.ndarray | list[int] | None = None,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Leave-one-condition-slice-out CP cross-validation.

    Holds out the entire slice `tensor[c, :, :, :]`, fits CP on the
    remaining conditions to learn B, C, D, and fits a_c for the held-out
    condition via linear regression against the Khatri-Rao product of
    (B (x) C (x) D).
    """
    from tensorly.tenalg import khatri_rao

    n_cond = tensor.shape[0]
    cond_indices = range(n_cond) if conditions is None else [int(c) for c in conditions]

    rows = []
    for held_out in cond_indices:
        train_idx = [i for i in range(n_cond) if i != held_out]
        sub_tensor = tensor[train_idx]

        w, f = parafac(
            tensor=sub_tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            normalize_factors=True,
            linesearch=True,
            random_state=random_state,
        )
        _, B, C, D = f

        # Form the design matrix: (B (x) C (x) D) has shape (n_send * n_rec * n_lr, rank)
        design = khatri_rao([B, C, D])  # (I1*I2*I3, rank)
        design = design * w[np.newaxis, :]

        # Flatten the held-out condition slice to match Khatri-Rao column-major mode ordering
        y = tensor[held_out].reshape(-1)

        a_c, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred_flat = design @ a_c

        sse = float(np.sum((y - pred_flat) ** 2))
        sst = float(np.sum(y**2))
        held_out_r2x = 1.0 - sse / sst if sst > 0 else np.nan

        rows.append(
            {
                "Rank": rank,
                "HeldOutCondition": held_out,
                "HeldOutConditionR2X": held_out_r2x,
                "TrainR2X": calculate_r2x(w, f, sub_tensor),
            }
        )

    return pd.DataFrame(rows)


def permute_lr_within_fiber(
    tensor: np.ndarray, random_state: int | None = None
) -> np.ndarray:
    """Permute LR-pair values within each (condition, sender, receiver) fiber.

    Destroys the low-rank structure of the LR mode while preserving both the
    marginal distribution of LR values per fiber and the non-negativity /
    sparsity geometry.
    """
    rng = np.random.default_rng(random_state)
    out = tensor.copy()
    orig_shape = out.shape
    flat = out.reshape(-1, orig_shape[-1])
    rand = rng.random(flat.shape)
    order = np.argsort(rand, axis=-1)
    flat = np.take_along_axis(flat, order, axis=-1)
    return flat.reshape(orig_shape)


def permute_eigenstate_labels(
    tensor: np.ndarray, random_state: int | None = None
) -> np.ndarray:
    """Independently permute sender and receiver eigen-state indices per condition."""
    rng = np.random.default_rng(random_state)
    out = tensor.copy()
    n_cond, n_send, n_rec, _ = out.shape
    for c in range(n_cond):
        p_send = rng.permutation(n_send)
        p_rec = rng.permutation(n_rec)
        out[c] = out[c][p_send][:, p_rec]
    return out


def cpd_permutation_null_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    n_null_replicates: int = 2,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Observed vs null R2X across ranks for both permutation nulls."""
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
        obs_r2x = calculate_r2x(w, f, tensor)
        rows.append(
            {"Rank": rank, "NullType": "observed", "Replicate": 0, "R2X": obs_r2x}
        )

        for rep in range(n_null_replicates):
            seed = None if random_state is None else random_state * 1000 + rep

            null_lr = permute_lr_within_fiber(tensor, random_state=seed)
            w_n, f_n = parafac(
                tensor=null_lr,
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
                    "R2X": calculate_r2x(w_n, f_n, null_lr),
                }
            )

            null_states = permute_eigenstate_labels(tensor, random_state=seed)
            w_s, f_s = parafac(
                tensor=null_states,
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
                    "NullType": "eigenstate_permutation",
                    "Replicate": rep,
                    "R2X": calculate_r2x(w_s, f_s, null_states),
                }
            )

    return pd.DataFrame(rows)


def corcondia_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Core consistency diagnostic (CORCONDIA; Bro & Kiers 2003) across ranks."""

    rows = []
    float(np.sum(tensor**2))
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
        r2x = calculate_r2x(w, f, tensor)

        # Scale factor 0 with the component weights so factors multiply to the CP tensor
        A, B, C, D = f[0] * w[np.newaxis, :], f[1], f[2], f[3]

        # Orthonormalize each factor matrix via pseudo-inverse for the projection
        A_pinv = np.linalg.pinv(A)
        B_pinv = np.linalg.pinv(B)
        C_pinv = np.linalg.pinv(C)
        D_pinv = np.linalg.pinv(D)

        # Core tensor G = tensor x_1 A+ x_2 B+ x_3 C+ x_4 D+
        # Using mode-by-mode matrix products:
        G = np.tensordot(tensor, A_pinv.T, axes=([0], [0]))  # (I2, I3, I4, R)
        G = np.tensordot(G, B_pinv.T, axes=([0], [0]))  # (I3, I4, R, R)
        G = np.tensordot(G, C_pinv.T, axes=([0], [0]))  # (I4, R, R, R)
        G = np.tensordot(G, D_pinv.T, axes=([0], [0]))  # (R, R, R, R)

        # G has indices corresponding to the order: (D_mode, C_mode, B_mode, A_mode) -> transpose to (A, B, C, D)
        G = np.moveaxis(G, [0, 1, 2, 3], [3, 2, 1, 0])

        # Target super-diagonal identity core T where T_{r,r,r,r} = 1
        T = np.zeros((rank, rank, rank, rank))
        for r in range(rank):
            T[r, r, r, r] = 1.0

        sse_core = float(np.sum((G - T) ** 2))
        corcondia = 100.0 * (1.0 - sse_core / rank)

        rows.append(
            {
                "Rank": rank,
                "CORCONDIA": corcondia,
                "R2X": r2x,
                "SSE_Core": sse_core,
            }
        )

    return pd.DataFrame(rows)


def cpd_component_significance(
    tensor: np.ndarray,
    rank: int,
    n_null: int = 20,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int = 0,
) -> pd.DataFrame:
    """Per-component permutation test against the fiber-permuted null.

    At a fixed rank, compares each component's raw singular value / weight
    w_r to the distribution of max(w) under the LR-within-fiber null,
    yielding a familywise-error-rate-controlled empirical p-value per component.
    """
    w_obs, _f_obs = parafac(
        tensor=tensor,
        rank=rank,
        n_iter_max=n_iter_max,
        init=init,
        normalize_factors=True,
        linesearch=True,
        random_state=random_state,
    )
    # Sort observed components descending by weight
    order = np.argsort(w_obs)[::-1]
    w_sorted = w_obs[order]

    null_max_weights = []
    null_weights_all = []
    for b in range(n_null):
        null_t = permute_lr_within_fiber(tensor, random_state=random_state * 1000 + b)
        w_null, _ = parafac(
            null_t,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            normalize_factors=True,
            linesearch=True,
            random_state=random_state,
        )
        null_max_weights.append(np.max(w_null))
        null_weights_all.append(np.sort(w_null)[::-1])

    null_max_weights = np.array(null_max_weights)
    null_weights_all = np.array(null_weights_all)

    rows = []
    total_var = np.sum(w_sorted**2)
    for r in range(rank):
        w_r = w_sorted[r]
        p_val = float((np.sum(null_max_weights >= w_r) + 1) / (n_null + 1))
        p_marginal = float((np.sum(null_weights_all[:, r] >= w_r) + 1) / (n_null + 1))
        rows.append(
            {
                "Rank": rank,
                "Component": r,
                "Weight": float(w_r),
                "ExplainedVarFrac": float(w_r**2 / total_var) if total_var > 0 else 0.0,
                "EmpiricalP": p_val,
                "EmpiricalP_marginal": p_marginal,
            }
        )

    return pd.DataFrame(rows)


def resample_tensor_conditions(
    tensor: np.ndarray, random_state: int | None = None
) -> np.ndarray:
    """Bootstrap conditions (patients) with replacement."""
    rng = np.random.default_rng(random_state)
    n_cond = tensor.shape[0]
    idx = rng.choice(n_cond, size=n_cond, replace=True)
    return tensor[idx].copy()


def split_half_conditions(
    tensor: np.ndarray, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split conditions (patients) into two disjoint halves."""
    rng = np.random.default_rng(random_state)
    n_cond = tensor.shape[0]
    perm = rng.permutation(n_cond)
    half = n_cond // 2
    return tensor[perm[:half]].copy(), tensor[perm[half:]].copy()


def cpd_replication_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    resample_axis: str = "lr",
    n_boot: int = 3,
    n_iter_max: int = 1000,
    init: str = "random",
    random_state: int | None = None,
    thresholds: tuple[float, ...] = (0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """Replicable-component counts under resampling of different tensor axes.

    `resample_axis` options:
      - 'lr': bootstrap LR pairs with replacement (tests robustness to LR panel)
      - 'conditions': bootstrap patients with replacement
      - 'splithalf_conditions': split patients into disjoint halves (tests cross-patient generalizability)
    """
    rows = []
    rng = np.random.default_rng(random_state)
    for rank in ranks:
        for b in range(n_boot):
            seed = None if random_state is None else random_state * 1000 + b
            if resample_axis == "lr":
                n_lr = tensor.shape[-1]
                idx = rng.choice(n_lr, size=n_lr, replace=True)
                boot_t = tensor[:, :, :, idx]
                _w_full, f_full = parafac(
                    tensor=tensor,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=random_state,
                )
                _w_b, f_b = parafac(
                    tensor=boot_t,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=seed,
                )
                D_boot_reindexed = np.zeros_like(f_full[3])
                for col_i, orig_i in enumerate(idx):
                    D_boot_reindexed[orig_i] += f_b[3][col_i]
                counts = component_replication_counts(
                    f_full[3], D_boot_reindexed, thresholds
                )
            elif resample_axis == "conditions":
                boot_t = resample_tensor_conditions(tensor, random_state=seed)
                _w_full, f_full = parafac(
                    tensor=tensor,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=random_state,
                )
                _w_b, f_b = parafac(
                    tensor=boot_t,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=seed,
                )
                counts = component_replication_counts(f_full[3], f_b[3], thresholds)
            elif resample_axis == "splithalf_conditions":
                half_a, half_b = split_half_conditions(tensor, random_state=seed)
                _, f_a = parafac(
                    tensor=half_a,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=seed,
                )
                _, f_b = parafac(
                    tensor=half_b,
                    rank=rank,
                    n_iter_max=n_iter_max,
                    init=init,
                    normalize_factors=True,
                    linesearch=True,
                    random_state=seed,
                )
                counts = component_replication_counts(f_a[3], f_b[3], thresholds)
            else:
                raise ValueError(f"Unknown resample_axis: {resample_axis!r}")

            rows.append(
                {"Rank": rank, "ResampleAxis": resample_axis, "Boot": b, **counts}
            )

    return pd.DataFrame(rows)


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
    of both R2X and the D-factor subspace.
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
    """2-D grid of held-out imputation error over (RISE rank x CPD rank)."""
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
    """Cross-validated AUC of the condition factor A at each CPD rank."""
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
