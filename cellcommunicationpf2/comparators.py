"""Python re-implementations of comparator methods used to benchmark CCC-RISE.

These are independent re-implementations, built to reproduce a method's
published algorithm closely enough for a fair comparison against CCC-RISE
without needing that method's original runtime (e.g. an R installation).
They are NOT wrappers around the original packages, and are not guaranteed
to be numerically identical to them -- see each function's docstring for the
specific source it was reimplemented from and any known differences.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot
from tensorly.tucker_tensor import tucker_to_tensor


@dataclass
class StaccatoResult:
    """Fitted values from :func:`staccato`.

    Attributes
    ----------
    W1, W2, W3, W4 : np.ndarray
        Orthonormal factor matrices for modes 1-4 (covariate, LR-pair,
        sender celltype, receiver celltype), each ``(mode_size, core_shape[i])``.
        ``W2``/``W3``/``W4`` are returned as ``pd.DataFrame`` (index = the
        supplied names) when the corresponding ``*_names`` argument is given,
        else as plain ``np.ndarray``.
    G : np.ndarray
        Core tensor of shape ``core_shape``, from the second (orthogonalizing)
        Tucker decomposition.
    fitted : np.ndarray
        Model-implied signal tensor, same shape as the input tensor:
        ``coef_tensor`` mode-multiplied by the original (unrotated) covariate
        matrices. Analogous to upstream's ``U``.
    coef_tensor : np.ndarray
        The covariate-coefficient tensor (upstream's ``C_ts``), shape
        ``(p1, d2, d3, d4)`` where ``p1`` is the covariate's column count.
    sigma2 : float
        Mean squared residual, ``mean((tensor - fitted) ** 2)``.
    log_likelihood : float
        Total (summed) Gaussian log-likelihood of the fit under i.i.d. noise
        with variance ``sigma2``. Upstream returns the full elementwise
        log-likelihood array; this returns its sum for convenience.
    """

    W1: np.ndarray
    W2: np.ndarray | pd.DataFrame
    W3: np.ndarray | pd.DataFrame
    W4: np.ndarray | pd.DataFrame
    G: np.ndarray
    fitted: np.ndarray
    coef_tensor: np.ndarray
    sigma2: float
    log_likelihood: float


def staccato(
    tensor: np.ndarray,
    core_shape: Sequence[int],
    covariate: np.ndarray | None = None,
    lr_names: Sequence[str] | None = None,
    sender_names: Sequence[str] | None = None,
    receiver_names: Sequence[str] | None = None,
    random_state: int | None = None,
) -> StaccatoResult:
    """Re-implementation of STACCato's supervised Tucker decomposition.

    This is a from-scratch Python/TensorLy re-implementation of the algorithm
    in ``STACCato:::staccato()`` (daiqile96/STACCato, R), reconstructed by
    reading that function's decompiled body -- it is not a wrapper around the
    R package, and STACCato is not a dependency of this project. It is meant
    to let CCC-RISE be benchmarked against STACCato's method without an R
    runtime (see ``R/README.md`` for the R comparator harness this replaces).

    The upstream algorithm fits a Tucker decomposition where each mode may be
    restricted to the column space of a user-supplied covariate/design
    matrix rather than a freely-estimated orthonormal factor (a supervised,
    or "envelope", Tucker model). Given per-mode covariate matrices
    ``X_1..X_4`` (``X_k = I`` for an unsupervised mode), it:

    1. QR-decomposes each ``X_k = Q_k R_k``.
    2. Rotates the tensor into the ``Q``-orthonormal basis and fits an
       ordinary (unsupervised) Tucker decomposition there at ``core_shape``.
    3. Un-rotates the reconstructed tensor by ``R_k^{-1}`` to get a
       covariate-coefficient tensor.
    4. Re-decomposes that coefficient tensor with a second Tucker fit, purely
       to obtain orthonormal, directly-interpretable factor matrices and a
       core tensor.
    5. Reconstructs the covariate-driven fitted tensor by mode-multiplying
       the coefficient tensor by the original (unrotated) covariate matrices.

    Only mode 1 (typically samples) accepts a supplied covariate here, matching
    how this project's R harness called STACCato (``run_staccato.R``): modes
    2-4 (LR-pair, sender, receiver) are always unsupervised (``X_k = I``), so
    ``core_shape[1:]`` behaves like an ordinary Tucker rank on those modes.

    Comparing against CCC-RISE
    ---------------------------
    STACCato is Tucker, not CP: its four factor matrices don't share a rank
    and aren't tied together component-by-component the way CCC-RISE's CP
    factors (``A``/``B``/``C``/``D``, see ``ccc_rise.py``) are. Reviewer #1
    specifically asked for an ALAD-status AUC comparison, not a factor-level
    concordance score, so the two intended comparisons are:

    1. **Downstream ALAD-status AUC (the reviewer's ask).** Feed this
       function's per-sample mode-1 scores -- ``covariate @ result.W1`` when
       ``covariate`` was supplied, or ``result.W1`` directly when it wasn't
       -- into the same logistic-regression/AUC pipeline already used to
       score CCC-RISE's condition factor ``A`` (see ``logreg.py``), and
       report both AUCs side by side.
    2. **Per-mode factor congruence (optional, for the concordance figure).**
       Build the input tensor the same way for both methods -- run
       ``ccc_rise.calc_communication_score_pseudobulk`` /
       ``pseudobulk_cp_decomposition`` on the same exported dataset used
       here, so both sides decompose an (as close as possible) identical
       sender x receiver x LR-pair x sample/condition tensor. Reindex
       ``W2``/``W3``/``W4`` (LR-pair, sender, receiver -- already
       label-indexed via ``lr_names``/``sender_names``/``receiver_names``)
       against CCC-RISE's ``D``/``B``/``C`` on shared labels, compute a
       cosine/Tucker congruence-coefficient matrix per mode between the two
       methods' columns, and solve the assignment with
       ``scipy.optimize.linear_sum_assignment`` to report each STACCato
       component's best-matching CCC-RISE component. Do this per mode
       independently -- Tucker's core tensor mixes components across modes,
       so there is no single shared component index to compare like there is
       between two CP decompositions.

    Parameters
    ----------
    tensor : np.ndarray
        4D array, shape ``(n_samples, n_lr_pairs, n_senders, n_receivers)``.
    core_shape : Sequence[int]
        Target Tucker rank per mode, length 4.
    covariate : np.ndarray, optional
        Mode-1 design matrix, shape ``(n_samples, p1)``. Defaults to
        ``np.eye(n_samples)`` (unsupervised mode 1), matching upstream's
        ``X_covar1 = NULL`` behavior.
    lr_names, sender_names, receiver_names : Sequence[str], optional
        Row labels for ``W2``/``W3``/``W4``. When given, that factor is
        returned as a ``pd.DataFrame`` indexed by these names instead of a
        plain array.
    random_state : int, optional
        Passed to both internal Tucker fits (``tensorly.decomposition.tucker``).

    Returns
    -------
    StaccatoResult
    """
    core_shape = tuple(core_shape)
    if len(core_shape) != 4 or tensor.ndim != 4:
        raise ValueError(
            f"staccato expects a 4-mode tensor and a 4-element core_shape, "
            f"got tensor.shape={tensor.shape} and core_shape={core_shape}"
        )
    for size, rank in zip(tensor.shape, core_shape):
        if rank > size:
            raise ValueError(
                f"core_shape {core_shape} exceeds tensor shape {tensor.shape}"
            )

    if covariate is None:
        covariate = np.eye(tensor.shape[0])
    covariates = [covariate, np.eye(tensor.shape[1]), np.eye(tensor.shape[2]), np.eye(tensor.shape[3])]

    Qs: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    for X in covariates:
        Q, R = np.linalg.qr(X)
        Qs.append(Q)
        Rs.append(R)

    rotated = multi_mode_dot(tensor, [Q.T for Q in Qs], modes=[0, 1, 2, 3])
    core_rot, factors_rot = tucker(
        rotated, rank=list(core_shape), init="svd", random_state=random_state
    )
    est_rot = tucker_to_tensor((core_rot, factors_rot))

    coef_tensor = multi_mode_dot(
        est_rot, [np.linalg.inv(R) for R in Rs], modes=[0, 1, 2, 3]
    )

    G, (W1, W2, W3, W4) = tucker(
        coef_tensor, rank=list(core_shape), init="svd", random_state=random_state
    )

    fitted = multi_mode_dot(coef_tensor, covariates, modes=[0, 1, 2, 3])
    residual = tensor - fitted
    sigma2 = float(np.mean(residual**2))
    log_likelihood = float(
        np.sum(-0.5 * np.log(2 * np.pi * sigma2) - residual**2 / (2 * sigma2))
    )

    def _label(factor: np.ndarray, names: Sequence[str] | None) -> np.ndarray | pd.DataFrame:
        if names is None:
            return factor
        return pd.DataFrame(factor, index=list(names))

    return StaccatoResult(
        W1=W1,
        W2=_label(W2, lr_names),
        W3=_label(W3, sender_names),
        W4=_label(W4, receiver_names),
        G=G,
        fitted=fitted,
        coef_tensor=coef_tensor,
        sigma2=sigma2,
        log_likelihood=log_likelihood,
    )
