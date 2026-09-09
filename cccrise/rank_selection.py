"""
Rank-selection criteria for RISE (PARAFAC2) and interaction-tensor (CPD) stages.
"""

from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
from scipy.sparse import issparse
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from .import_data import add_cond_idxs

Factors = list[np.ndarray]


# =============================================================================
# Helper utilities
# =============================================================================


def calculate_r2x(weights: np.ndarray, factors: Factors, tensor: np.ndarray) -> float:
    """Calculate reconstruction R2X for a CP decomposition.

    Parameters
    ----------
    weights : np.ndarray
        Component weights from the CP decomposition.
    factors : list[np.ndarray]
        List of factor matrices from the CP decomposition.
    tensor : np.ndarray
        Original data tensor against which reconstruction is evaluated.

    Returns
    -------
    float
        Fraction of variance explained (R2X) by the CP decomposition.
    """
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

    `parafac2_nd` centers the data using `var["means"]`; if genes are dropped
    without recomputing this column, centering becomes inconsistent with the
    sliced data.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression data object.
    gene_mask : np.ndarray
        Boolean mask of shape (n_vars,) indicating which genes to retain.

    Returns
    -------
    anndata.AnnData
        A subsetted copy of the AnnData object with recomputed `var["means"]`.
    """
    sub = adata[:, gene_mask].copy()
    X = cast(Any, sub.X)
    sub.var["means"] = np.asarray(X.mean(axis=0)).ravel()
    return sub


def _projections_to_matrix(
    adata: anndata.AnnData, projections: list[np.ndarray]
) -> np.ndarray:
    """Assemble a per-condition projection list into a single (n_obs, rank) matrix.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression data containing 'condition_unique_idxs' in `obs`.
    projections : list[np.ndarray]
        List of per-condition projection matrices.

    Returns
    -------
    np.ndarray
        Combined cell projection matrix of shape (n_obs, rank).
    """
    sg_index = adata.obs["condition_unique_idxs"].to_numpy().astype(int)
    rank = projections[0].shape[1]
    out = np.zeros((adata.n_obs, rank))
    for i, p in enumerate(projections):
        out[sg_index == i, :] = p
    return out


def _cell_loadings(
    weights: np.ndarray,
    factors: Factors,
    projections: np.ndarray,
    condition_idx: np.ndarray,
) -> np.ndarray:
    """Per-cell RISE loadings -- the design a held-out gene block is predicted from.

    The PARAFAC2 model is ``X_i = P_i B diag(w * a_i) C^T``, so a cell's loading
    is its row of ``P_i B`` scaled by its condition's ``w * a_i``. The order is
    load-bearing: ``B`` is a dense rank x rank matrix, so scaling ``P_i`` before
    multiplying by ``B`` is a different model whenever ``a_i`` varies across
    conditions -- which is the entire point of the ``A`` factor.
    """
    A, B, _C = factors
    return (projections @ B) * (A[condition_idx] * weights)


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

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression data.
    rank : int
        Decomposition rank to evaluate.
    condition_column : str
        Column name in `adata.obs` specifying sample or condition.
    gene_test_frac : float, default=0.1
        Fraction of genes held out for testing.
    n_iter_max : int, default=100
        Maximum number of iterations for PARAFAC2 fitting.
    tol : float, default=1e-6
        Tolerance threshold for PARAFAC2 convergence.
    random_state : int | None, default=None
        Random seed for gene and cell splitting and model initialization.

    Returns
    -------
    dict[str, float]
        Dictionary with fold results containing:
        - 'Rank': Evaluated rank.
        - 'HeldOutBlockR2X': Reconstruction R2X on the held-out test block (G_te x N_b).
        - 'TrainBlockR2X': In-sample R2X on the training block (G_tr, all cells).
        - 'NTestGenes': Number of test genes in G_te.
        - 'NTrainGenes': Number of training genes in G_tr.
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

    (weights, factors, projections), train_r2x_reported = parafac2_nd(
        train_genes,
        rank=rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
    )
    P = _projections_to_matrix(train_genes, projections)

    sg_index = adata.obs["condition_unique_idxs"].to_numpy(dtype=int)
    n_cells = adata.n_obs
    cell_half = rng.integers(0, 2, size=n_cells).astype(bool)  # True -> N_b

    # Shared by the least-squares fit of C_te and the prediction step.
    L = _cell_loadings(weights, factors, P, sg_index)  # (n_cells, rank)

    X_te = adata.X[:, te_gene_idx]
    X_te_genes = X_te.toarray() if issparse(X_te) else np.asarray(X_te)

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
    rows = [
        bicv_reconstruction_error(
            adata=adata,
            rank=rank,
            condition_column=condition_column,
            gene_test_frac=gene_test_frac,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=fold,
        )
        | {"Fold": fold}
        for rank in ranks
        for fold in range(n_folds)
    ]
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
    """Calculate one fold of entry-masked CP cross-validation reconstruction error.

    A random `mask_frac` of tensor entries are held out (mask=0), CP is fit
    with `tensorly`'s built-in EM-style masked ALS, and R2X is scored on the
    held-out entries only using the same normalization as `calculate_r2x`
    (denominator is the sum of squares of the held-out raw values).

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor to decompose.
    rank : int
        Decomposition rank to evaluate.
    mask_frac : float, default=0.1
        Fraction of tensor entries randomly held out for testing.
    n_iter_max : int, default=1000
        Maximum number of iterations for CP decomposition.
    init : str, default="random"
        Initialization method for CP decomposition (e.g., 'random', 'svd').
    random_state : int | None, default=None
        Random seed for mask generation and CP initialization.

    Returns
    -------
    dict[str, float]
        Dictionary with evaluation results containing:
        - 'Rank': Evaluated rank.
        - 'HeldOutR2X': Reconstruction R2X on held-out tensor entries.
        - 'InSampleR2X': Reconstruction R2X on the full tensor.
        - 'NHeldOut': Total count of held-out entries.
        - 'NTrain': Total count of training entries.
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
    """Entry-masked cross-validation swept across ranks and mask replicates.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor to decompose.
    ranks : list[int]
        List of ranks to evaluate.
    n_masks : int, default=5
        Number of random mask replicates per rank.
    mask_frac : float, default=0.1
        Fraction of tensor entries held out in each mask.
    n_iter_max : int, default=1000
        Maximum number of iterations for CP decomposition.
    init : str, default="random"
        Initialization method for CP decomposition.
    random_state : int | None, default=None
        Base random seed for reproducibility across replicates.

    Returns
    -------
    pd.DataFrame
        DataFrame containing results for each rank and replicate with columns:
        'Rank', 'Mask', 'HeldOutR2X', 'InSampleR2X', 'NHeldOut', 'NTrain'.
    """
    rows = [
        cp_masked_reconstruction_error(
            tensor,
            rank=rank,
            mask_frac=mask_frac,
            n_iter_max=n_iter_max,
            init=init,
            random_state=None if random_state is None else random_state * 1000 + m,
        )
        | {"Mask": m}
        for rank in ranks
        for m in range(n_masks)
    ]
    return pd.DataFrame(rows)
