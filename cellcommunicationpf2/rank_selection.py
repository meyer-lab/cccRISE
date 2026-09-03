"""
Rank-selection criteria for RISE (PARAFAC2) and interaction-tensor (CPD) stages.
"""

from typing import Any, cast

import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
from scipy.sparse import issparse
from tensorly.base import unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import mode_dot  # ty: ignore[unresolved-import]

from .ccc_rise import cp_decomposition
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

    `anndata_to_list` centers each condition's data using `var["means"]`; if genes
    are dropped without recomputing this column, centering becomes inconsistent
    with the sliced data.

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

    (_, factors, projections), train_r2x_reported = parafac2_nd(
        train_genes,
        rank=rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
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


def cp_fit_r2x_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    n_iter_max: int = 100,
    tol: float = 1e-3,
    init: str = "svd",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Full-tensor CP fit R2X swept across ranks.

    Nothing is held out: every rank is fit against all entries, so R2X rises
    monotonically and measures how much structure a rank can absorb.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor to decompose.
    ranks : list[int]
        List of ranks to evaluate.
    n_iter_max : int, default=100
        Maximum number of ALS iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    init : str, default="svd"
        Initialization method for CP decomposition.
    random_state : int | None, default=None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'Rank' and 'FitR2X'.
    """
    rows = []
    for rank in ranks:
        _weights, _factors, r2x = cp_decomposition(
            tensor,
            rank,
            n_iter_max=n_iter_max,
            tol=tol,
            init=init,
            random_state=random_state,
        )
        rows.append({"Rank": rank, "FitR2X": r2x})

    return pd.DataFrame(rows)


# =============================================================================
# Heuristic: component-collinearity / degeneracy screen
# =============================================================================


def cpd_component_collinearity(
    factors_by_rank: dict[int, tuple[np.ndarray, Factors]],
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Screen CPD factor models across ranks for degenerate ("canceling") pairs.

    For every rank, computes the signed cosine similarity between every pair
    of components within each of the 4 factor modes (condition, sender,
    receiver, LR pair). A pair that is highly collinear (|cosine| >
    `threshold`) in several modes at once, with an odd number of those modes
    negatively signed, is the classic CPD degeneracy signature: two rank-1
    terms that are nearly identical in most modes but flip sign in one,
    nearly canceling in the sum despite each carrying substantial weight.
    A pair collinear across modes with no sign flip is a near-duplicate
    (redundant) rather than canceling pair.

    Parameters
    ----------
    factors_by_rank : dict[int, tuple[np.ndarray, Factors]]
        Maps CP rank to (weights, [A, B, C, D]).
    threshold : float, default=0.9
        Absolute cosine similarity above which two components are
        considered collinear in a given mode.

    Returns
    -------
    pd.DataFrame
        One row per rank with columns:
        - 'Rank'
        - 'MaxModesCollinear': largest number of modes (0-4) in which any
          component pair exceeds `threshold`.
        - 'WorstPairCanceling': whether the worst pair's collinear modes
          have a net-negative sign product (canceling) rather than
          net-positive (near-duplicate). NaN when 'MaxModesCollinear' is 0.
        - 'WorstPairI', 'WorstPairJ': 1-based component indices of the worst
          pair; -1 when rank < 2.
    """
    rows = []
    for rank in sorted(factors_by_rank):
        _weights, factors = factors_by_rank[rank]
        if rank < 2:
            rows.append(
                {
                    "Rank": rank,
                    "MaxModesCollinear": 0,
                    "WorstPairCanceling": np.nan,
                    "WorstPairI": -1,
                    "WorstPairJ": -1,
                }
            )
            continue

        cos_by_mode = []
        for f in factors:
            norm = f / np.linalg.norm(f, axis=0, keepdims=True)
            cos_by_mode.append(norm.T @ norm)

        best_count = -1
        best_pair = (-1, -1)
        best_signs: list[float] = []
        for i in range(rank):
            for j in range(i + 1, rank):
                signs = [cos_by_mode[m][i, j] for m in range(4)]
                count = int(sum(abs(s) > threshold for s in signs))
                if count > best_count:
                    best_count = count
                    best_pair = (i + 1, j + 1)
                    best_signs = signs

        canceling = np.nan
        if best_count > 0:
            collinear_signs = [s for s in best_signs if abs(s) > threshold]
            canceling = bool(np.prod(np.sign(collinear_signs)) < 0)

        rows.append(
            {
                "Rank": rank,
                "MaxModesCollinear": best_count,
                "WorstPairCanceling": canceling,
                "WorstPairI": best_pair[0],
                "WorstPairJ": best_pair[1],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Heuristic: core consistency diagnostic (CORCONDIA, generalized to 4 modes)
# =============================================================================


def cp_core_consistency(
    tensor: np.ndarray, weights: np.ndarray, factors: Factors
) -> float:
    """Core consistency diagnostic (CORCONDIA), generalized to 4 modes.

    Projects the tensor onto each factor matrix's pseudoinverse to obtain
    the best-fit core in the CP factors' own basis, then compares that core
    to the ideal super-diagonal core a valid quadrilinear CP model implies.
    Consistency near 100 means the rank-R CP model is an appropriate,
    essentially unique decomposition; it typically drops sharply once rank
    exceeds the tensor's true multilinear rank, since additional components
    start trading off against each other (the pseudoinverse projection
    picks up large off-diagonal core entries) rather than each explaining
    independent structure.

    `tensor` may equally be a CANDELINC core with `factors` in that core's
    own space: the bases are orthonormal, so ``pinv(U_n F_n) = pinv(F_n)
    U_n^T`` and the projected core comes out identical either way. On a
    cohort-scale tensor that is the difference between projecting ~300M
    entries and ~600k, so `cp_core_consistency_sweep` is normally handed the
    compressed core.

    Parameters
    ----------
    tensor : np.ndarray
        4D communication tensor the factors were fit to, or the CANDELINC
        core of it when `factors` are core-space factors.
    weights : np.ndarray
        Component weights, shape (rank,).
    factors : list[np.ndarray]
        The 4 CP factor matrices, each shape (dim_n, rank).

    Returns
    -------
    float
        Core consistency, in percent (100 = perfectly super-diagonal core;
        can go negative for a badly-fitting rank).
    """
    rank = factors[0].shape[1]
    scaled_factors = [factors[0] * weights[np.newaxis, :], *factors[1:]]
    pinvs = [np.linalg.pinv(f) for f in scaled_factors]

    core = tensor
    for mode, pinv in enumerate(pinvs):
        core = mode_dot(core, pinv, mode)

    ideal = np.zeros((rank,) * 4)
    diag_idx = np.arange(rank)
    ideal[diag_idx, diag_idx, diag_idx, diag_idx] = 1.0

    return float(100.0 * (1.0 - np.sum((core - ideal) ** 2) / np.sum(ideal**2)))


def cp_core_consistency_sweep(
    tensor: np.ndarray, factors_by_rank: dict[int, tuple[np.ndarray, Factors]]
) -> pd.DataFrame:
    """Core consistency across every rank in a set of CP fits.

    Parameters
    ----------
    tensor : np.ndarray
        4D communication tensor the fits' factors were fit to.
    factors_by_rank : dict[int, tuple[np.ndarray, Factors]]
        Maps CP rank to (weights, [A, B, C, D]).

    Returns
    -------
    pd.DataFrame
        Columns 'Rank' and 'CoreConsistency'.
    """
    rows = [
        {
            "Rank": rank,
            "CoreConsistency": cp_core_consistency(tensor, weights, factors),
        }
        for rank, (weights, factors) in sorted(factors_by_rank.items())
    ]
    return pd.DataFrame(rows)


# =============================================================================
# CANDELINC compression: fit CP inside a small orthonormal subspace
# =============================================================================


def candelinc_compress(
    tensor: np.ndarray, max_rank: int, oversample: int = 5
) -> tuple[np.ndarray, Factors, float]:
    """Compress a tensor to a CANDELINC core sized for ranks up to `max_rank`.

    CANDELINC (Carroll, Pruzansky & Kruskal 1980) fits CP under the constraint
    that each factor matrix lies in the column space of a fixed basis. Taking
    those bases from a truncated HOSVD turns CP-ALS on the full tensor into
    CP-ALS on a core small enough to sweep many ranks over, then maps the
    solution back with ``A_n = U_n F_n``.

    Both rank-selection metrics are invariant to the compression as long as
    each basis keeps at least `max_rank` columns. The bases are orthonormal,
    so within-mode cosines between decompressed components equal those between
    core components (`cpd_component_collinearity`), and ``pinv(U_n F_n) =
    pinv(F_n) U_n^T`` makes the projected core identical whether it is taken
    from the original tensor or from `core` (`cp_core_consistency`). Only fit
    error changes, and only by the fixed amount that lies outside the retained
    subspaces -- which `candelinc_r2x` adds back.

    Modes are compressed in order of how much they shrink, so every mode after
    the first works on an already smaller tensor. Each mode keeps
    ``min(dim_n, max_rank + oversample)`` leading left singular vectors of its
    unfolding, computed through the mode's Gram matrix (cheap here -- the
    sender, receiver and LR modes of a communication tensor are at most a few
    thousand wide).

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to compress; any number of modes.
    max_rank : int
        Largest CP rank the core must support. Each basis keeps at least this
        many columns, which is what makes the metrics compression-invariant.
    oversample : int, default=5
        Extra basis columns beyond `max_rank`, giving CP-ALS room to rotate
        within the subspace instead of pinning it to exactly `max_rank`
        directions.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], float]
        The core, the per-mode orthonormal bases, and `lost_sq` -- the squared
        norm the projection discarded, which any CP model of the core carries
        as an irreducible error floor against the original tensor.

    Raises
    ------
    ValueError
        If `max_rank` exceeds the smallest mode of `tensor`, which would make
        the core unable to represent a rank-`max_rank` model.
    """
    if max_rank > min(tensor.shape):
        raise ValueError(
            f"max_rank={max_rank} exceeds the smallest mode of the tensor "
            f"(shape {tuple(tensor.shape)}). Compression cannot support a rank "
            f"the tensor itself cannot."
        )

    keep = [min(dim, max_rank + oversample) for dim in tensor.shape]
    # Compress the most-compressible modes first: every later Gram matrix is
    # then formed from an already smaller tensor.
    order = sorted(range(tensor.ndim), key=lambda m: keep[m] / tensor.shape[m])

    norm_sq = float(np.sum(tensor**2))
    bases: Factors = [np.empty(0)] * tensor.ndim
    core = tensor
    for mode in order:
        if keep[mode] == core.shape[mode]:
            bases[mode] = np.eye(core.shape[mode])
            continue
        unfolded = unfold(core, mode)
        gram = unfolded @ unfolded.T
        # `eigh` returns ascending eigenvalues, so the leading singular
        # directions are the trailing columns. Their order within the basis
        # is irrelevant: CANDELINC only depends on the span.
        _eigvals, eigvecs = np.linalg.eigh(gram)
        basis = eigvecs[:, -keep[mode] :]
        bases[mode] = basis
        core = mode_dot(core, basis.T, mode)

    lost_sq = max(0.0, norm_sq - float(np.sum(core**2)))
    return core, bases, lost_sq


def candelinc_r2x(core_r2x: float, core: np.ndarray, lost_sq: float) -> float:
    """Restate a core-space R2X as R2X against the original tensor.

    A CP model fit inside a CANDELINC core reports its error against the core,
    which omits everything the projection discarded. Against the original
    tensor the error is ``lost_sq + ||core - CP(core)||^2``.

    Parameters
    ----------
    core_r2x : float
        R2X the fit reported against `core`.
    core : np.ndarray
        The CANDELINC core the fit was run on.
    lost_sq : float
        Squared norm discarded by the compression, from `candelinc_compress`.

    Returns
    -------
    float
        R2X against the uncompressed tensor.
    """
    core_sq = float(np.sum(core**2))
    norm_sq = core_sq + lost_sq
    if norm_sq == 0:
        return 0.0
    return float(1.0 - (lost_sq + (1.0 - core_r2x) * core_sq) / norm_sq)
