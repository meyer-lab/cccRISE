"""
Rank-selection criteria for RISE (PARAFAC2) and interaction-tensor (CPD) stages.
"""

from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
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
