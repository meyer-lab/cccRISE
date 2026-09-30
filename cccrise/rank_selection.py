import numpy as np
import pandas as pd
from tensorly.base import unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import mode_dot  # ty: ignore[unresolved-import]

Factors = list[np.ndarray]


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


def cpd_component_collinearity(
    factors_by_rank: dict[int, tuple[np.ndarray, Factors]],
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Screen CPD factor models across ranks for degenerate ("canceling") pairs.

    Two rank-1 terms that are nearly identical in most modes but flip sign in
    one nearly cancel in the sum, despite each carrying substantial weight.
    A pair collinear across modes with no sign flip is a near-duplicate rather
    than a canceling pair, so the sign is reported separately from the count.

    Returns one row per rank:
    ``MaxModesCollinear``, ``WorstPairCanceling``, and
    ``WorstPairI`` / ``WorstPairJ`` (1-based component indices, -1 when
    rank < 2).
    """
    rows = []
    for rank in sorted(factors_by_rank):
        _weights, factors = factors_by_rank[rank]
        if rank < 2:
            rows.append(_no_collinearity_row(rank))
            continue

        best_count, canceling, best_pair = _worst_collinear_pair(
            factors, rank, threshold
        )
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


def _no_collinearity_row(rank: int) -> dict:
    return {
        "Rank": rank,
        "MaxModesCollinear": 0,
        "WorstPairCanceling": np.nan,
        "WorstPairI": -1,
        "WorstPairJ": -1,
    }


def _worst_collinear_pair(
    factors: Factors, rank: int, threshold: float
) -> tuple[int, float, tuple[int, int]]:
    """Find the component pair with the most modes exceeding the collinearity
    threshold, and whether its collinear modes cancel (net-negative sign)."""
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

    return best_count, canceling, best_pair


def cp_core_consistency(
    tensor: np.ndarray, weights: np.ndarray, factors: Factors
) -> float:
    """Core consistency diagnostic (CORCONDIA), generalized to 4 modes.

    Projects `tensor` onto each factor matrix's pseudoinverse to get the
    best-fit core in the CP factors' own basis, then compares that core to the
    super-diagonal core a valid quadrilinear model implies. Near 100 means the
    rank-R model is appropriate and essentially unique. It drops sharply once
    rank exceeds the number of components a valid CP model supports, because the extra
    components start trading off against one another and the projection picks
    up large off-diagonal entries.

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
    (`cp_core_consistency`).

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
        # directions are the trailing columns.
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
