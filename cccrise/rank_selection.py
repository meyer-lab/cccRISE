"""Rank-selection criteria for the RISE (PARAFAC2) and CPD stages.

RISE ranks are chosen by bi-cross-validation (`run_bicv`); CPD ranks by core
consistency (`cp_core_consistency_sweep`) and a component-collinearity screen
(`cpd_component_collinearity`), both read off one CP fit per rank.
"""

from typing import Any, NamedTuple, cast

import anndata
import numpy as np
import pandas as pd
from parafac2.parafac2 import parafac2_nd
from scipy.sparse import csr_array, issparse
from tensorly.base import unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import mode_dot  # ty: ignore[unresolved-import]

from .import_data import add_cond_idxs

Factors = list[np.ndarray]

#: Nonzeros per row-chunk in the streaming passes over a cell x gene matrix.
#: 2e8 nonzeros is ~1.6 GB as float32 values plus int32 column indices, which
#: leaves room on a 24 GB device for the dense operands and temporaries.
DEFAULT_CHUNK_NNZ = 200_000_000


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
# Streaming primitives for a cell x gene matrix
# =============================================================================
#
# A cohort-scale expression matrix does not fit on a GPU (the IBDverse terminal
# ileum block is 3.6e9 nonzeros, ~44 GB), so every pass below streams it in row
# chunks, uploading one chunk at a time. Each primitive is written so that a
# gene subset is expressed by *zeroing rows of the dense operand* rather than
# by slicing the sparse matrix: slicing 90% of the columns out of a 44 GB CSR
# costs a 40 GB copy per fold, while carrying the other 10% of the nonzeros
# through the multiply costs ~11% more arithmetic on a step that is transfer
# bound anyway.


def _has_cuda() -> bool:
    """Whether CuPy is installed and a CUDA device is visible."""
    try:
        import cupy  # ty: ignore[unresolved-import]

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _use_gpu(device: str | None) -> bool:
    """Resolve a device request to whether the GPU path is taken.

    None auto-detects; an explicit 'cuda' without CuPy or a CUDA device falls
    back to the CPU rather than raising.
    """
    if device is None:
        device = "cuda" if _has_cuda() else "cpu"
    return device.startswith("cuda") and _has_cuda()


def _as_canonical_csr(matrix: Any) -> Any:
    """The matrix as a canonical CSR, or unchanged if it is dense.

    Both `_row_block` and cuSPARSE rely on sorted, duplicate-free indices. The
    check caches on the matrix, so re-wrapping a matrix that is already CSR is
    avoided: a fresh wrapper would drop the cached flag and rescan every
    nonzero on every pass.
    """
    if not issparse(matrix):
        return matrix
    if matrix.format != "csr":
        matrix = csr_array(matrix)
    if not matrix.has_canonical_format:
        matrix.sum_duplicates()
    return matrix


def _row_block(matrix: Any, start: int, stop: int) -> Any:
    """Rows ``[start, stop)``, sharing the parent's value and index arrays.

    SciPy's own row slicing copies both, which at cohort scale is ~1.6 GB of
    memcpy per chunk on a step that is otherwise transfer bound. Slicing the
    CSR arrays directly is a view; only the (tiny) row pointer is rebuilt.
    Expects a matrix from `_as_canonical_csr`.
    """
    if not issparse(matrix):
        return matrix[start:stop]

    lo, hi = int(matrix.indptr[start]), int(matrix.indptr[stop])
    block = csr_array(
        (
            matrix.data[lo:hi],
            matrix.indices[lo:hi],
            matrix.indptr[start : stop + 1] - lo,
        ),
        shape=(stop - start, matrix.shape[1]),
    )
    # Every row block of a canonical matrix is itself canonical.
    block.has_canonical_format = True
    return block


def _row_chunks(matrix: Any, chunk_nnz: int) -> list[tuple[int, int]]:
    """Split rows into contiguous chunks of roughly `chunk_nnz` nonzeros."""
    n_rows = matrix.shape[0]
    if n_rows == 0:
        return []
    nnz = matrix.nnz if issparse(matrix) else matrix.size
    per_row = max(1.0, nnz / n_rows)
    rows = max(1, min(n_rows, int(chunk_nnz / per_row)))
    return [(s, min(s + rows, n_rows)) for s in range(0, n_rows, rows)]


def _to_device(block: Any, use_gpu: bool) -> Any:
    """Move one row chunk to the compute device, keeping its storage format.

    CSR chunks are rebuilt with int32 index arrays: cuSPARSE requires them,
    and a chunk is sized so that both its column indices and its row pointers
    fit regardless of the parent matrix's index width.
    """
    if not use_gpu:
        return block

    import cupy  # ty: ignore[unresolved-import]

    if not issparse(block):
        return cupy.asarray(block)

    import cupyx.scipy.sparse as cusparse  # ty: ignore[unresolved-import]

    device_block = cusparse.csr_matrix(
        (
            cupy.asarray(block.data),
            cupy.asarray(block.indices.astype(np.int32)),
            cupy.asarray(block.indptr.astype(np.int32)),
        ),
        shape=block.shape,
    )
    # cuSPARSE's spmm refuses a non-canonical CSR rather than fixing one, and a
    # matrix built from raw index arrays carries no canonical flag. `_row_block`
    # guarantees the host chunk is canonical, so this resolves to a ~3 ms
    # device-side check rather than the COO round-trip `sum_duplicates` does on
    # unsorted input.
    device_block.sum_duplicates()
    return device_block


def _xp(use_gpu: bool) -> Any:
    """The array module for the resolved device."""
    if use_gpu:
        import cupy  # ty: ignore[unresolved-import]

        return cupy

    return np


def _to_host(array: Any) -> np.ndarray:
    """Bring a device array back to the host; a no-op for NumPy input."""
    return cast(np.ndarray, array.get()) if hasattr(array, "get") else np.asarray(array)


def _centered_matmul(
    matrix: Any,
    dense: np.ndarray,
    means: np.ndarray,
    use_gpu: bool,
    chunk_nnz: int = DEFAULT_CHUNK_NNZ,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """``(matrix - 1 means^T) @ dense``, streamed over row chunks.

    Zero rows in `dense` (and the matching zeros in `means`) restrict the
    product to a gene subset without touching the sparse matrix.
    """
    xp = _xp(use_gpu)
    matrix = _as_canonical_csr(matrix)
    n_rows, _ = matrix.shape
    width = dense.shape[1]
    if out is None:
        out = np.empty((n_rows, width))

    dense_dev = xp.asarray(dense, dtype=xp.float32)
    shift = xp.asarray(means, dtype=xp.float64) @ xp.asarray(dense, dtype=xp.float64)

    for start, stop in _row_chunks(matrix, chunk_nnz):
        block = _to_device(_row_block(matrix, start, stop), use_gpu)
        product = xp.asarray(block @ dense_dev, dtype=xp.float64) - shift
        out[start:stop] = _to_host(product)

    return out


def _transposed_product(block: Any, left: np.ndarray, use_gpu: bool) -> Any:
    """``block^T @ left`` for one chunk, without transposing `block`.

    CuPy's ``dense @ sparse`` goes through `sum_duplicates`, which round-trips
    the chunk through COO and allocates several times the chunk's own size --
    fatal at cohort scale. cuSPARSE's spmm takes the transpose as a flag
    instead, so nothing is reformatted.
    """
    if not use_gpu:
        return block.T @ left

    import cupy  # ty: ignore[unresolved-import]
    import cupyx.cusparse  # ty: ignore[unresolved-import]
    import cupyx.scipy.sparse as cusparse  # ty: ignore[unresolved-import]

    left = cupy.asfortranarray(cupy.asarray(left, dtype=block.dtype))
    # `scipy.sparse.issparse` is False for a CuPy sparse matrix, so the device
    # check has to be CuPy's own.
    if not cusparse.issparse(block):
        return block.T @ left

    return cupyx.cusparse.spmm(block, left, transa=True)


def _centered_rmatmul(
    dense: np.ndarray,
    matrix: Any,
    means: np.ndarray,
    use_gpu: bool,
    chunk_nnz: int = DEFAULT_CHUNK_NNZ,
) -> np.ndarray:
    """``dense^T @ (matrix - 1 means^T)``, streamed over row chunks.

    Zero rows in `dense` restrict the product to a cell subset, the way zero
    rows in `_centered_matmul`'s operand restrict it to a gene subset.
    """
    xp = _xp(use_gpu)
    matrix = _as_canonical_csr(matrix)
    width = dense.shape[1]
    total = xp.zeros((matrix.shape[1], width), dtype=xp.float64)
    column_weight = xp.zeros(width, dtype=xp.float64)

    for start, stop in _row_chunks(matrix, chunk_nnz):
        block = _to_device(_row_block(matrix, start, stop), use_gpu)
        left = dense[start:stop]
        total += xp.asarray(_transposed_product(block, left, use_gpu), dtype=xp.float64)
        column_weight += xp.asarray(left, dtype=xp.float64).sum(axis=0)

    total -= xp.outer(xp.asarray(means, dtype=xp.float64), column_weight)
    return np.ascontiguousarray(_to_host(total).T)


def _column_moments(
    matrix: Any,
    use_gpu: bool,
    row_mask: np.ndarray | None = None,
    chunk_nnz: int = DEFAULT_CHUNK_NNZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column sums and sums of squares of the raw (uncentered) matrix.

    Restricted to `row_mask`'s rows when given. Combined with the column means
    these give the centered squared norm of any gene subset in closed form --
    ``sum_g (sq_g - n_rows * mean_g^2)`` -- so nothing has to re-traverse the
    matrix to learn what a compression discarded or what a held-out block's
    total sum of squares is.
    """
    xp = _xp(use_gpu)
    matrix = _as_canonical_csr(matrix)
    n_genes = matrix.shape[1]
    sums = xp.zeros(n_genes, dtype=xp.float64)
    squares = xp.zeros(n_genes, dtype=xp.float64)
    sparse = issparse(matrix)

    for start, stop in _row_chunks(matrix, chunk_nnz):
        block = _to_device(_row_block(matrix, start, stop), use_gpu)
        weight = None
        if row_mask is not None:
            weight = xp.asarray(row_mask[start:stop], dtype=xp.float64)

        if sparse:
            values = xp.asarray(block.data, dtype=xp.float64)
            if weight is not None:
                values = values * xp.repeat(weight, xp.diff(block.indptr))
            sums += xp.bincount(block.indices, weights=values, minlength=n_genes)
            squares += xp.bincount(
                block.indices,
                weights=values * xp.asarray(block.data, dtype=xp.float64),
                minlength=n_genes,
            )
        else:
            dense = xp.asarray(block, dtype=xp.float64)
            if weight is not None:
                dense = dense * weight[:, None]
                squares += xp.sum(dense * xp.asarray(block, dtype=xp.float64), axis=0)
            else:
                squares += xp.sum(dense**2, axis=0)
            sums += xp.sum(dense, axis=0)

    return _to_host(sums), _to_host(squares)


def _select_columns(
    matrix: Any, columns: np.ndarray, use_gpu: bool, chunk_nnz: int = DEFAULT_CHUNK_NNZ
) -> Any:
    """``matrix[:, columns]``, built one row chunk at a time.

    Only used for the held-out gene block, which is a small fraction of the
    matrix. The training block is never sliced -- see the module note above.
    """
    if not issparse(matrix):
        return np.asarray(matrix)[:, columns]

    from scipy.sparse import vstack

    matrix = _as_canonical_csr(matrix)
    chunks = _row_chunks(matrix, chunk_nnz)
    if len(chunks) == 1:
        return csr_array(matrix[:, columns])
    return csr_array(
        vstack([_row_block(matrix, a, b)[:, columns] for a, b in chunks], format="csr")
    )


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


# =============================================================================
# CANDELINC compression for the RISE (PARAFAC2) stage
# =============================================================================


class RiseCore(NamedTuple):
    """A CANDELINC-compressed RISE problem, fittable at any rank up to its size.

    Attributes
    ----------
    core : anndata.AnnData
        The compressed slabs stacked into one small AnnData, carrying the
        `condition_unique_idxs` and all-zero `var['means']` that
        `parafac2.parafac2_nd` expects. Already centered, hence the zero means.
    cell_basis : np.ndarray
        Shape (n_obs, n_basis); condition ``i``'s rows hold the orthonormal
        basis ``Q_i`` that maps its core-space projections back to its cells.
        Columns past `basis_widths[i]` are zero.
    condition_idx : np.ndarray
        Shape (n_obs,); which condition each row of `cell_basis` belongs to.
    basis_widths : np.ndarray
        Columns of `cell_basis` each condition actually uses -- ``n_basis``
        except for a condition with fewer cells than that.
    gene_basis : np.ndarray
        Shape (n_genes, n_basis); the retained gene subspace, with zero rows
        for genes excluded from the fit.
    lost_sq : float
        Squared norm the gene-mode projection discarded, in the units of the
        original (centered) matrix.
    """

    core: anndata.AnnData
    cell_basis: np.ndarray
    condition_idx: np.ndarray
    basis_widths: np.ndarray
    gene_basis: np.ndarray
    lost_sq: float


def rise_candelinc_compress(
    adata: anndata.AnnData,
    condition_column: str,
    max_rank: int,
    gene_mask: np.ndarray | None = None,
    oversample: int = 10,
    n_power_iter: int = 2,
    device: str | None = None,
    random_state: int | None = None,
    column_squares: np.ndarray | None = None,
    chunk_nnz: int = DEFAULT_CHUNK_NNZ,
) -> RiseCore:
    """Compress a RISE problem to per-condition cores sized for `max_rank`.

    Two compressions, only the first of which loses anything.

    **Gene mode (CANDELINC, lossy).** Constrain ``C`` to a fixed ``n_basis``
    dimensional gene subspace ``W``. Because every reconstruction
    ``P_i diag(a_i) B C^T`` then lies in ``W``'s row space, the loss splits
    exactly into an in-subspace part and the fixed residual ``lost_sq`` that
    lies outside it, so fitting the projected slabs ``X_i W`` minimises the
    original objective subject to ``C = W C_core``. This is the PARAFAC2
    analogue of `candelinc_compress`, and `candelinc_r2x` restates a core-space
    R2X against the original matrix the same way.

    **Cell mode (exact).** The optimal ``P_i`` for a slab lies in that slab's
    own column space, so writing ``X_i W = Q_i S_i`` and substituting
    ``P_i = Q_i P_core_i`` -- still orthonormal, since ``Q_i`` is -- leaves the
    loss unchanged while shrinking condition ``i`` from ``cells_i x n_genes``
    to at most ``n_basis x n_basis``. This is what makes the compressed sweep
    essentially free: the IBDverse cohort's 1.19M x 23.7k matrix becomes 343
    slabs of 60 x 60, and every rank in a sweep refits that instead of the
    original.

    Neither step looks at anything outside `gene_mask`, so a compression built
    from a BiCV fold's training genes stays disjoint from that fold's held-out
    genes, exactly as the RISE fit it replaces does.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell expression data. ``var['means']`` supplies the centering.
    condition_column : str
        Column in `adata.obs` identifying the sample or condition.
    max_rank : int
        Largest RISE rank the core must support.
    gene_mask : np.ndarray, optional
        Boolean mask of genes to fit on; None uses every gene.
    oversample : int, default=10
        Basis columns beyond `max_rank`, giving the fit room to rotate inside
        the subspace rather than pinning it to exactly `max_rank` directions.
    n_power_iter : int, default=2
        Power iterations in the randomized range finder that builds `W`. Each
        one is two streaming passes and sharpens the retained subspace toward
        the leading singular directions.
    device : str, optional
        'cuda', 'cpu', or None to use a GPU when CuPy and CUDA are available.
    random_state : int, optional
        Seed for the range finder's test matrix.
    column_squares : np.ndarray, optional
        Precomputed per-gene sums of squares from `_column_moments`. Depends on
        neither the rank nor the gene mask, so a sweep computes it once and
        hands it to every fold.
    chunk_nnz : int
        Nonzeros per streaming row chunk.

    Returns
    -------
    RiseCore
        The compressed problem.
    """
    use_gpu = _use_gpu(device)
    adata = add_cond_idxs(adata, condition_column)
    matrix = cast(Any, adata.X)
    n_obs, n_genes = adata.shape

    mask = np.ones(n_genes, dtype=bool) if gene_mask is None else np.asarray(gene_mask)
    condition_idx = adata.obs["condition_unique_idxs"].to_numpy(dtype=int)
    counts = np.bincount(condition_idx, minlength=int(condition_idx.max()) + 1)
    n_basis = int(min(max_rank + oversample, mask.sum()))
    if n_basis < max_rank:
        raise ValueError(
            f"max_rank={max_rank} exceeds the {int(mask.sum())} genes available "
            "to fit on; the core cannot represent a model that large."
        )

    means = np.asarray(adata.var["means"], dtype=np.float64) * mask

    # --- Gene mode: randomized range finder for the retained subspace --------
    rng = np.random.default_rng(random_state)
    sketch = rng.standard_normal((n_genes, n_basis)) * mask[:, None]
    sample = _centered_matmul(matrix, sketch, means, use_gpu, chunk_nnz)
    for _ in range(n_power_iter):
        left, _ = np.linalg.qr(sample)
        gene_block = _centered_rmatmul(left, matrix, means, use_gpu, chunk_nnz).T
        gene_block *= mask[:, None]
        gene_dirs, _ = np.linalg.qr(gene_block)
        sample = _centered_matmul(
            matrix, gene_dirs * mask[:, None], means, use_gpu, chunk_nnz, out=sample
        )

    left, _ = np.linalg.qr(sample)
    gene_block = _centered_rmatmul(left, matrix, means, use_gpu, chunk_nnz) * mask
    _u, _sv, vt = np.linalg.svd(gene_block, full_matrices=False)
    gene_basis = (vt[:n_basis] * mask).T

    # --- Cell mode: per-condition QR of the projected slabs ------------------
    # `projected` is overwritten in place with the Q factors: the projected
    # slab is not needed once its QR is taken, and at cohort scale the buffer
    # is large enough to be worth not duplicating.
    projected = _centered_matmul(matrix, gene_basis, means, use_gpu, chunk_nnz)
    widths = np.minimum(counts, n_basis)
    slabs = []
    core_conditions = []
    for i, width in enumerate(widths):
        rows = np.flatnonzero(condition_idx == i)
        basis, upper = np.linalg.qr(projected[rows])
        padded = np.zeros((rows.size, n_basis))
        padded[:, :width] = basis[:, :width]
        projected[rows] = padded
        slabs.append(upper[:width])
        core_conditions.extend([i] * int(width))

    if column_squares is None:
        _sums, column_squares = _column_moments(matrix, use_gpu, chunk_nnz=chunk_nnz)
    norm_sq = float(np.sum((column_squares - n_obs * means**2) * mask))
    lost_sq = max(0.0, norm_sq - float(sum(np.sum(s**2) for s in slabs)))

    core = anndata.AnnData(
        X=np.concatenate(slabs, axis=0),
        obs=pd.DataFrame({"condition_unique_idxs": core_conditions}).astype(int),
        var=pd.DataFrame({"means": np.zeros(n_basis)}),
    )
    return RiseCore(core, projected, condition_idx, widths, gene_basis, lost_sq)


def rise_candelinc_fit(
    compressed: RiseCore,
    rank: int,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> tuple[np.ndarray, Factors, np.ndarray, float]:
    """Fit RISE at `rank` inside a `RiseCore`, decompressed back to full size.

    Parameters
    ----------
    compressed : RiseCore
        Output of `rise_candelinc_compress`.
    rank : int
        RISE rank; must not exceed the core's basis size.
    n_iter_max, tol
        PARAFAC2 fitting controls.
    random_state : int, optional
        Seed for the fit.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], np.ndarray, float]
        Weights, factors ``[A, B, C]`` with ``C`` mapped back to gene space,
        the per-cell projection matrix ``(n_obs, rank)``, and R2X restated
        against the original uncompressed matrix.
    """
    (weights, factors, core_projections), core_r2x = parafac2_nd(
        compressed.core,
        rank=rank,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
    )

    basis = compressed.cell_basis
    projections = np.zeros((basis.shape[0], rank))
    for i, width in enumerate(compressed.basis_widths):
        rows = np.flatnonzero(compressed.condition_idx == i)
        projections[rows] = basis[rows, : int(width)] @ core_projections[i]

    factors = [factors[0], factors[1], compressed.gene_basis @ factors[2]]
    r2x = candelinc_r2x(core_r2x, cast(Any, compressed.core.X), compressed.lost_sq)
    return weights, factors, projections, r2x


def _bicv_fold_split(
    adata: anndata.AnnData, gene_test_frac: float, random_state: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one BiCV fold's gene split and cell split.

    Returns the training-gene mask, the held-out gene indices, and the boolean
    mask selecting the N_b cells (True) from the N_a cells (False).
    """
    rng = np.random.default_rng(random_state)
    n_genes = adata.n_vars
    n_test_genes = max(1, round(n_genes * gene_test_frac))
    gene_perm = rng.permutation(n_genes)
    test_genes = gene_perm[:n_test_genes]

    train_mask = np.zeros(n_genes, dtype=bool)
    train_mask[gene_perm[n_test_genes:]] = True
    return train_mask, test_genes, rng.integers(0, 2, size=adata.n_obs).astype(bool)


def _cell_loadings(
    weights: np.ndarray,
    factors: Factors,
    projections: np.ndarray,
    condition_idx: np.ndarray,
) -> np.ndarray:
    """Per-cell RISE loadings: the design the held-out gene block is predicted from.

    The PARAFAC2 model is ``X_i = P_i B diag(w * a_i) C^T``, so a cell's
    loading vector is its row of ``P_i B``, scaled by its condition's
    ``w * a_i``. The order matters: ``B`` is a dense rank x rank matrix, so
    scaling ``P_i`` before multiplying by ``B`` is a different (and wrong)
    model whenever ``a_i`` varies across conditions.
    """
    A, B, _C = factors
    return (projections @ B) * (A[condition_idx] * weights)


def _held_out_block_r2x(
    test_block: Any,
    na_mask: np.ndarray,
    nb_mask: np.ndarray,
    loadings: np.ndarray,
    moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    use_gpu: bool,
    chunk_nnz: int = DEFAULT_CHUNK_NNZ,
) -> float:
    """Refit the held-out genes' loadings on N_a and score them on N_b.

    Every quantity is accumulated through streaming products against the
    sparse held-out block: a cohort-scale test block is ~20 GB dense, and
    materialising it is both unnecessary and, at IBDverse scale, fatal.
    Restricting a product to N_a or N_b is done by zeroing the other half's
    rows of `loadings` rather than by slicing the block.

    Parameters
    ----------
    test_block : Any
        The held-out genes' raw (uncentered) expression, shape (n_obs, n_test).
    na_mask, nb_mask : np.ndarray
        Complementary boolean cell masks: N_a fits ``C_te``, N_b scores it.
    loadings : np.ndarray
        Per-cell loadings from `_cell_loadings`, shape (n_obs, rank).
    moments : tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(sums over N_a, sums over N_b, squares over N_b)`` from
        `_column_moments`. Rank-independent, so a sweep computes them once
        per fold.
    use_gpu : bool
        Whether to stream through the GPU.
    chunk_nnz : int
        Nonzeros per streaming row chunk.

    Returns
    -------
    float
        R2X on the G_te x N_b block.
    """
    rank = loadings.shape[1]
    sums_a, sums_b, squares_b = moments
    n_a = int(na_mask.sum())
    n_b = int(nb_mask.sum())
    mean_a = sums_a / n_a

    # Centering on N_a's mean keeps the held-out block disjoint from N_b.
    sst = float(np.sum(squares_b - 2.0 * mean_a * sums_b + n_b * mean_a**2))
    if sst <= 0:
        return float("nan")

    split_loadings = np.concatenate(
        [loadings * na_mask[:, None], loadings * nb_mask[:, None]], axis=1
    )
    cross = _centered_rmatmul(split_loadings, test_block, mean_a, use_gpu, chunk_nnz)
    cross_a, cross_b = cross[:rank], cross[rank:]
    gram_a = split_loadings[:, :rank].T @ split_loadings[:, :rank]
    gram_b = split_loadings[:, rank:].T @ split_loadings[:, rank:]

    # Normal equations rather than `lstsq` on the (n_obs, rank) design: the
    # design is never materialised as a centered dense block, and `lstsq` on
    # the rank x rank system keeps the minimum-norm behaviour for a
    # rank-deficient fit.
    c_te = np.linalg.lstsq(gram_a, cross_a, rcond=None)[0]

    sse = (
        sst
        - 2.0 * float(np.sum(c_te * cross_b))
        + float(np.sum(c_te * (gram_b @ c_te)))
    )
    return 1.0 - sse / sst


def _bicv_fold(
    adata: anndata.AnnData,
    ranks: list[int],
    condition_column: str,
    gene_test_frac: float,
    n_iter_max: int,
    tol: float,
    random_state: int | None,
    compress: bool,
    oversample: int,
    n_power_iter: int,
    device: str | None,
    chunk_nnz: int,
    column_squares: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Score every rank in `ranks` against a single BiCV fold.

    The fold's split, its CANDELINC compression and its held-out block are all
    built once and shared by every rank, which is what makes a sweep cost
    roughly one fold's data traversal rather than one per (rank, fold) pair.
    """
    use_gpu = _use_gpu(device)
    adata = add_cond_idxs(adata, condition_column)
    condition_idx = adata.obs["condition_unique_idxs"].to_numpy(dtype=int)
    matrix = cast(Any, adata.X)

    train_mask, test_genes, cell_half = _bicv_fold_split(
        adata, gene_test_frac, random_state
    )
    # The score sums over held-out genes, so their order is irrelevant --
    # and selecting them in sorted order leaves the block canonical, which is
    # what the GPU's sparse products want.
    test_genes = np.sort(test_genes)
    na_mask, nb_mask = ~cell_half, cell_half

    test_block = _select_columns(matrix, test_genes, use_gpu, chunk_nnz)
    sums_a, _ = _column_moments(test_block, use_gpu, na_mask, chunk_nnz)
    sums_b, squares_b = _column_moments(test_block, use_gpu, nb_mask, chunk_nnz)
    moments = (sums_a, sums_b, squares_b)

    compressed = None
    train_genes_adata = None
    if compress:
        compressed = rise_candelinc_compress(
            adata,
            condition_column,
            max_rank=max(ranks),
            gene_mask=train_mask,
            oversample=oversample,
            n_power_iter=n_power_iter,
            device=device,
            random_state=random_state,
            column_squares=column_squares,
            chunk_nnz=chunk_nnz,
        )
    else:
        train_genes_adata = _subset_genes_with_recomputed_means(adata, train_mask)

    rows = []
    for rank in ranks:
        if compressed is not None:
            weights, factors, projections, train_r2x = rise_candelinc_fit(
                compressed, rank, n_iter_max, tol, random_state
            )
        else:
            assert train_genes_adata is not None
            (weights, factors, projection_list), train_r2x = parafac2_nd(
                train_genes_adata,
                rank=rank,
                n_iter_max=n_iter_max,
                tol=tol,
                random_state=random_state,
            )
            projections = _projections_to_matrix(train_genes_adata, projection_list)

        loadings = _cell_loadings(weights, factors, projections, condition_idx)
        rows.append(
            {
                "Rank": rank,
                "HeldOutBlockR2X": _held_out_block_r2x(
                    test_block, na_mask, nb_mask, loadings, moments, use_gpu, chunk_nnz
                ),
                "TrainBlockR2X": train_r2x,
                "NTestGenes": int(test_genes.size),
                "NTrainGenes": int(train_mask.sum()),
            }
        )
    return rows


def bicv_reconstruction_error(
    adata: anndata.AnnData,
    rank: int,
    condition_column: str,
    gene_test_frac: float = 0.1,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    random_state: int | None = None,
    compress: bool = True,
    oversample: int = 10,
    n_power_iter: int = 2,
    device: str | None = None,
    chunk_nnz: int = DEFAULT_CHUNK_NNZ,
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

    CANDELINC compression does not touch that bookkeeping: the compression is
    built from G_tr alone, so it stands in for the RISE fit it replaces and is
    equally disjoint from G_te. See `rise_candelinc_compress`.

    Note that this scores a rank in isolation, so its core is sized for `rank`
    alone. `run_bicv` shares one core across the whole sweep and therefore
    sizes it for the largest rank, which makes its row for a given rank use a
    wider basis than this function would. The two agree to well inside fold
    noise -- widening the basis is what `oversample` does, and the criterion is
    insensitive to it -- but they are not bit-identical.

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
    compress : bool, default=True
        Fit inside a CANDELINC core instead of on the full training block.
    oversample : int, default=10
        Basis columns beyond `rank` the core keeps when compressing.
    n_power_iter : int, default=2
        Power iterations in the compression's randomized range finder.
    device : str, optional
        'cuda', 'cpu', or None to use a GPU when CuPy and CUDA are available.
    chunk_nnz : int
        Nonzeros per streaming row chunk.

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
    return _bicv_fold(
        adata,
        [rank],
        condition_column,
        gene_test_frac=gene_test_frac,
        n_iter_max=n_iter_max,
        tol=tol,
        random_state=random_state,
        compress=compress,
        oversample=oversample,
        n_power_iter=n_power_iter,
        device=device,
        chunk_nnz=chunk_nnz,
    )[0]


def run_bicv(
    adata: anndata.AnnData,
    ranks: list[int] | range,
    condition_column: str,
    n_folds: int = 3,
    gene_test_frac: float = 0.1,
    n_iter_max: int = 100,
    tol: float = 1e-6,
    compress: bool = True,
    oversample: int = 10,
    n_power_iter: int = 2,
    device: str | None = None,
    chunk_nnz: int = DEFAULT_CHUNK_NNZ,
    fold_seed_offset: int = 0,
) -> pd.DataFrame:
    """Run bi-cross-validation rank selection across specified ranks and folds.

    Folds are the outer loop and ranks the inner one, so each fold's split,
    compression and held-out block are built once and reused by every rank.

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
    compress : bool, default=True
        CANDELINC-compress each fold once and fit every rank inside the
        resulting core. `False` refits the full training block per rank.
    oversample : int, default=10
        Basis columns beyond ``max(ranks)`` each core keeps.
    n_power_iter : int, default=2
        Power iterations in the compression's randomized range finder.
    device : str, optional
        'cuda', 'cpu', or None to use a GPU when CuPy and CUDA are available.
    chunk_nnz : int
        Nonzeros per streaming row chunk.
    fold_seed_offset : int, default=0
        Added to each fold's seed. Folds are otherwise seeded by index, so two
        sweeps of the same data draw identical blocks; shifting the offset draws
        a different fold structure instead, which is how the criterion's
        sensitivity to fold assignment is measured (a noise-floor question, not
        a resampling one).

    Returns
    -------
    pd.DataFrame
        DataFrame containing results for each rank and fold with columns:
        'Rank', 'Fold', 'HeldOutBlockR2X', 'TrainBlockR2X', 'NTestGenes', 'NTrainGenes'.
    """
    ranks = list(ranks)

    # Depends on neither the fold nor the rank, and costs a full pass, so it is
    # computed once here and handed to every fold's compression.
    column_squares = None
    if compress:
        _sums, column_squares = _column_moments(
            cast(Any, adata.X), _use_gpu(device), chunk_nnz=chunk_nnz
        )

    rows = [
        row | {"Fold": fold}
        for fold in range(n_folds)
        for row in _bicv_fold(
            adata,
            ranks,
            condition_column,
            gene_test_frac=gene_test_frac,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=fold + fold_seed_offset,
            compress=compress,
            oversample=oversample,
            n_power_iter=n_power_iter,
            device=device,
            chunk_nnz=chunk_nnz,
            column_squares=column_squares,
        )
    ]
    return pd.DataFrame(rows).sort_values(["Rank", "Fold"], ignore_index=True)


# =============================================================================
# CPD rank-selection criteria for the interaction tensor
# =============================================================================


def cpd_component_collinearity(
    factors_by_rank: dict[int, tuple[np.ndarray, Factors]],
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Screen CPD factor models across ranks for degenerate ("canceling") pairs.

    Two rank-1 terms that are nearly identical in most modes but flip sign in
    one nearly cancel in the sum, despite each carrying substantial weight.
    That is the classic CPD degeneracy signature, and it says the rank is too
    high more reliably than fit error does. A pair collinear across modes with
    *no* sign flip is a near-duplicate rather than a canceling pair, so the
    sign is reported separately from the count.

    Returns one row per rank:
    ``MaxModesCollinear`` (0-4, how many modes the worst pair exceeds
    `threshold` in), ``WorstPairCanceling`` (whether that pair's collinear
    modes have a net-negative sign product; NaN when the count is 0), and
    ``WorstPairI`` / ``WorstPairJ`` (1-based component indices, -1 when
    rank < 2).
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

        # Signed cosine between every component pair, within each mode.
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


def cp_core_consistency(
    tensor: np.ndarray, weights: np.ndarray, factors: Factors
) -> float:
    """Core consistency diagnostic (CORCONDIA), generalized to 4 modes.

    Projects `tensor` onto each factor matrix's pseudoinverse to get the
    best-fit core in the CP factors' own basis, then compares that core to the
    super-diagonal core a valid quadrilinear model implies. Near 100 means the
    rank-R model is appropriate and essentially unique; it drops sharply once
    rank exceeds the tensor's true multilinear rank, because the extra
    components start trading off against one another and the projection picks
    up large off-diagonal entries.

    `tensor` may equally be a CANDELINC core with `factors` in that core's own
    space: the bases are orthonormal, so ``pinv(U_n F_n) = pinv(F_n) U_n^T``
    and the projected core is identical either way. On a cohort-scale tensor
    that is the difference between projecting ~300M entries and ~600k, so the
    sweep normally passes the compressed core.

    Returned in percent; can go negative for a badly-fitting rank.
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

    Columns: 'Rank', 'CoreConsistency'.
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

    Both rank-selection metrics are invariant to this, as long as each basis
    keeps at least `max_rank` columns. The bases are orthonormal, so
    within-mode cosines between decompressed components equal those between
    core components (`cpd_component_collinearity`), and
    ``pinv(U_n F_n) = pinv(F_n) U_n^T`` makes the projected core identical
    whether taken from the original tensor or the core
    (`cp_core_consistency`). Only fit error changes, and only by the fixed
    amount lying outside the retained subspaces -- which `candelinc_r2x` adds
    back.

    Returns the core, the per-mode bases, and the squared norm discarded.

    Raises
    ------
    ValueError
        If `max_rank` exceeds the smallest mode of `tensor`, which would leave
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
    core = tensor
    bases: Factors = [np.empty(0)] * tensor.ndim
    for mode in order:
        if keep[mode] == core.shape[mode]:
            bases[mode] = np.eye(core.shape[mode])
            continue
        unfolded = unfold(core, mode)
        gram = unfolded @ unfolded.T
        # `eigh` returns ascending eigenvalues, so the leading singular
        # directions are the trailing columns. Their order within the basis is
        # irrelevant: CANDELINC depends only on the span.
        _eigvals, eigvecs = np.linalg.eigh(gram)
        bases[mode] = eigvecs[:, -keep[mode] :]
        core = mode_dot(core, bases[mode].T, mode)

    lost_sq = max(0.0, norm_sq - float(np.sum(core**2)))
    return core, bases, lost_sq


def candelinc_r2x(core_r2x: float, core: np.ndarray, lost_sq: float) -> float:
    """Restate a core-space R2X as R2X against the original tensor.

    A CP model fit inside a CANDELINC core reports error against the core,
    which omits everything the projection discarded. Against the original
    tensor the error is ``lost_sq + ||core - CP(core)||^2``.
    """
    core_sq = float(np.sum(core**2))
    norm_sq = core_sq + lost_sq
    if norm_sq == 0:
        return 0.0
    return float(1.0 - (lost_sq + (1.0 - core_r2x) * core_sq) / norm_sq)
