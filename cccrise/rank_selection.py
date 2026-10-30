from collections.abc import Hashable

import numpy as np
import pandas as pd
import tensorly as tl
from scipy.sparse.csgraph import connected_components
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

Factors = list[np.ndarray]

#: Precision every CP fit runs in unless a caller asks otherwise. On the ALAD
#: and IBD tensors, float32 fits stop earlier than float64 ones (the error
#: change falls below float32 resolution) but reach the same optima at the
#: selected ranks (per-component FMS 1.000 to the float64 CPU fits).
CP_DTYPE = "float32"
#: Precision of masked (imputation) fits: in float32 the EM fill-in on ALAD
#: went NaN at CP rank 8 (float64: imputation R2X 0.311), though it matched
#: float64 to 1e-4 at ranks 3 and 5.
MASKED_CP_DTYPE = "float64"
#: CP fit precisions and devices callers may choose (`fit_cp_on_device`).
CP_DTYPES = ("float32", "float64")
CP_DEVICES = ("gpu", "cpu")


def calculate_r2x(weights: np.ndarray, factors: Factors, tensor: np.ndarray) -> float:
    """Calculate reconstruction R2X for a CP decomposition.

    Uses ``||X - Xhat||^2 = ||X||^2 - 2<X, Xhat> + ||Xhat||^2``, contracting
    `tensor` with the factors one mode at a time, so the reconstruction (as
    large as `tensor`) is never built.

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
    flat = np.asarray(tensor).reshape(-1)
    norm_tensor_sq = float(np.dot(flat, flat))
    if norm_tensor_sq == 0:
        return 0.0
    weights = np.asarray(weights, dtype=float)
    factors = [np.asarray(f, dtype=float) for f in factors]
    inner = np.tensordot(tensor, factors[-1], axes=([tensor.ndim - 1], [0]))
    for factor in reversed(factors[:-1]):
        inner = np.einsum("...ir,ir->...r", inner, factor)
    gram = np.prod([f.T @ f for f in factors], axis=0)
    norm_residual_sq = (
        norm_tensor_sq - 2 * float(weights @ inner) + float(weights @ gram @ weights)
    )
    return float(1.0 - norm_residual_sq / norm_tensor_sq)


def max_mode_cosines(factors: Factors) -> list[float]:
    """Per mode, the largest |cosine| between two of the fit's components
    (NaN for a rank-1 fit): how collinear its most collinear pair is there."""
    cosines = []
    for f in factors:
        unit = f / np.linalg.norm(f, axis=0, keepdims=True)
        gram = np.abs(unit.T @ unit)
        np.fill_diagonal(gram, np.nan)
        cosines.append(float(np.nanmax(gram)) if f.shape[1] > 1 else np.nan)
    return cosines


def distinct_minima(
    start_fms: pd.DataFrame, r2x: pd.Series, same_minimum_fms: float = 0.95
) -> pd.DataFrame:
    """The distinct minima a rank's seeded starts reach.

    Two starts reach the same minimum when their FMS is at least
    `same_minimum_fms`; a minimum is a connected group of starts. `start_fms`
    holds one row per pair of starts ('Rank', 'SeedA', 'SeedB', 'FMS') and
    `r2x` each start's R2X, indexed by ('Rank', 'Seed').

    Returns one row per minimum: its 'Rank', the 'Starts' reaching it, its
    best 'R2X', the 'FMS' of its best start to the best start of the minimum
    most starts reach (f_max), and its 'Order' within the rank (1 = most
    starts; ties: higher R2X). 'Tied' marks ranks whose two most-reached
    minima tie, which have no f_max; their FMS is to the tied minimum with
    the higher R2X.
    """
    frames = [
        _rank_minima(rank, pairs, r2x.loc[rank], same_minimum_fms)
        for rank, pairs in start_fms.groupby("Rank")
    ]
    return pd.concat(frames, ignore_index=True)


def _rank_minima(
    rank: Hashable, pairs: pd.DataFrame, r2x: pd.Series, same_minimum_fms: float
) -> pd.DataFrame:
    seeds = sorted(set(pairs["SeedA"]) | set(pairs["SeedB"]))
    index = {seed: i for i, seed in enumerate(seeds)}
    a, b = pairs["SeedA"].map(index), pairs["SeedB"].map(index)
    fms = np.eye(len(seeds))
    fms[a, b] = fms[b, a] = pairs["FMS"]
    _, labels = connected_components(fms >= same_minimum_fms, directed=False)
    fit = r2x.loc[seeds].to_numpy()
    groups = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        best = members[fit[members].argmax()]
        groups.append((len(members), fit[best], best))
    groups.sort(key=lambda g: (-g[0], -g[1]))
    reference = groups[0][2]
    return pd.DataFrame(
        {
            "Rank": rank,
            "Starts": n,
            "R2X": best_r2x,
            "FMS": fms[best, reference],
            "Order": order,
            "Tied": len(groups) > 1 and groups[0][0] == groups[1][0],
        }
        for order, (n, best_r2x, best) in enumerate(groups, start=1)
    )


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


def cp_backend(device: str | None = None) -> str:
    """The tensorly backend a CP fit on `device` runs on: 'cupy' for 'gpu',
    'numpy' for 'cpu', and for None the GPU whenever CuPy can see one."""
    if device not in (None, *CP_DEVICES):
        raise ValueError(f"device must be one of {CP_DEVICES} or None; got {device!r}.")
    if device == "cpu" or (device is None and not _gpu_available()):
        return "numpy"
    _real_conj_on_cupy()
    return "cupy"


def _gpu_available() -> bool:
    try:
        import cupy  # ty: ignore[unresolved-import]

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no CuPy, or CuPy installed without a visible GPU
        return False


def _real_conj_on_cupy() -> None:
    """Make tensorly's CuPy `conj` the identity on real arrays.

    tensorly's randomized SVD (the SVD start) takes
    ``conj(transpose(A))`` of each unfolding, and CuPy's `conj` copies even a
    real array. On the IBD tensor that second copy of a 60 x 25.8M unfolding
    runs a 24 GB GPU out of memory; the conjugate of a real array is the
    array itself, so nothing numerical changes.
    """
    import cupy  # ty: ignore[unresolved-import]

    with tl.backend_context("cupy"):
        tl.backend.core.Backend._available_backends["cupy"].register_method(
            "conj",
            lambda x, *args, **kwargs: (
                cupy.conj(x, *args, **kwargs) if cupy.iscomplexobj(x) else x
            ),
        )


def fit_cp_on_device(
    tensor: np.ndarray,
    rank: int,
    device: str | None = None,
    dtype: str = CP_DTYPE,
    mask: np.ndarray | None = None,
    **parafac_kwargs,
) -> tuple[np.ndarray, Factors]:
    """`parafac` on `device` (see `cp_backend`; None: the GPU when there is
    one) in `dtype`, returning float64 NumPy weights and factors.

    `tensor` is cast on the host before it is moved, so the device never
    holds a float64 copy of a tensor fit in float32.
    """
    with tl.backend_context(cp_backend(device)):
        cp = parafac(
            tl.tensor(np.asarray(tensor, dtype=dtype)),
            rank,
            mask=None if mask is None else tl.tensor(np.asarray(mask, dtype=dtype)),
            **parafac_kwargs,
        )
        return (
            tl.to_numpy(cp.weights).astype(float),
            [tl.to_numpy(f).astype(float) for f in cp.factors],
        )


def masked_cp_reconstruction(
    tensor: np.ndarray,
    observed: np.ndarray,
    rank: int,
    device: str | None = None,
    dtype: str = MASKED_CP_DTYPE,
    **parafac_kwargs,
) -> np.ndarray:
    """CP reconstruction fit to the `observed` entries only, on `device` in
    `dtype` (see `fit_cp_on_device`).

    tensorly's masked ALS imputes unobserved entries with the current model
    each iteration (EM). They are zeroed first because its SVD init and first
    ALS sweep read them.
    """
    return cp_to_tensor(
        fit_cp_on_device(
            np.where(observed, tensor, 0.0),
            rank,
            device,
            dtype,
            observed,
            **parafac_kwargs,
        )
    )


def cp_imputation_r2x(
    tensor: np.ndarray,
    rank: int,
    n_folds: int = 10,
    random_state: int | None = None,
    device: str | None = None,
    dtype: str = MASKED_CP_DTYPE,
    **parafac_kwargs,
) -> float:
    """K-fold entry-masked imputation R2X of a rank-`rank` CP model.

    Entries are split into `n_folds` random folds; each fold is predicted by a
    CP fit to the others (`masked_cp_reconstruction`), and R2X pools every
    entry's prediction against the whole tensor, as in `calculate_r2x`.
    Pooling scores every entry exactly once, so the result doesn't hinge on
    which of a heavy-tailed tensor's few dominant entries a single random hold
    out happens to draw.
    """
    rng = np.random.default_rng(random_state)
    fold = (rng.permutation(tensor.size) % n_folds).reshape(tensor.shape)
    imputed = np.empty_like(tensor, dtype=float)
    for k in range(n_folds):
        held_out = fold == k
        imputed[held_out] = masked_cp_reconstruction(
            tensor,
            ~held_out,
            rank,
            device,
            dtype,
            random_state=random_state,
            **parafac_kwargs,
        )[held_out]
    return float(1.0 - np.sum((tensor - imputed) ** 2) / np.sum(tensor**2))


def cp_imputation_sweep(
    tensor: np.ndarray,
    ranks: list[int],
    n_repeats: int = 3,
    n_folds: int = 10,
    random_state: int = 0,
    device: str | None = None,
    dtype: str = MASKED_CP_DTYPE,
    **parafac_kwargs,
) -> pd.DataFrame:
    """`cp_imputation_r2x` at every rank, once per fold split.

    Repeat `r` uses the same split (seeded `random_state + r`) at every rank.
    Columns: 'Rank', 'Repeat', 'Seed', 'ImputationR2X'.
    """
    return pd.DataFrame(
        {
            "Rank": rank,
            "Repeat": r,
            "Seed": random_state + r,
            "ImputationR2X": cp_imputation_r2x(
                tensor,
                rank,
                n_folds,
                random_state + r,
                device,
                dtype,
                **parafac_kwargs,
            ),
        }
        for r in range(n_repeats)
        for rank in ranks
    )
