"""Python/TensorLy re-implementation of STACCato (daiqile96/STACCato, R).

See :func:`staccato`'s docstring for what algorithm this reproduces, how it
differs from the original R package, and how to compare its output against
cccRISE.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot  # ty: ignore[unresolved-import]
from tensorly.tucker_tensor import tucker_to_tensor


@dataclass
class StaccatoResult:
    """Fitted values from :func:`staccato`.

    Attributes
    ----------
    W1, W2, W3, W4 : np.ndarray
        Orthonormal factor matrices for modes 1-4 (covariate, LR-pair,
        sender celltype, receiver celltype). ``W1`` is ``(p1, core_shape[0])``,
        ``p1`` being the covariate's column count (``n_samples`` when
        unsupervised); ``W2``-``W4`` are ``(mode_size, core_shape[i])``.
        ``W2``/``W3``/``W4`` are returned as ``pd.DataFrame`` (index = the
        supplied names) when the corresponding ``*_names`` argument is given,
        else as plain ``np.ndarray``.
    G : np.ndarray
        Core tensor of shape ``core_shape``, from the second (orthogonalizing)
        Tucker decomposition.
    fitted : np.ndarray or None
        Model-implied signal tensor, same shape as the input tensor:
        ``coef_tensor`` mode-multiplied by the original (unrotated) covariate
        matrices. Analogous to upstream's ``U``. None unless `full_output`.
    coef_tensor : np.ndarray or None
        The covariate-coefficient tensor (upstream's ``C_ts``), shape
        ``(p1, d2, d3, d4)`` where ``p1`` is the covariate's column count.
        None unless `full_output`.
    sigma2 : float
        Mean squared residual, ``mean((tensor - fitted) ** 2)`` (NaN unless
        `full_output`).
    log_likelihood : float
        Total (summed) Gaussian log-likelihood of the fit under i.i.d. noise
        with variance ``sigma2``. Upstream returns the full elementwise
        log-likelihood array; this returns its sum for convenience (NaN
        unless `full_output`).
    """

    W1: np.ndarray
    W2: np.ndarray | pd.DataFrame
    W3: np.ndarray | pd.DataFrame
    W4: np.ndarray | pd.DataFrame
    G: np.ndarray
    fitted: np.ndarray | None
    coef_tensor: np.ndarray | None
    sigma2: float
    log_likelihood: float


def _mode_products(tensor: np.ndarray, matrices: Sequence[np.ndarray]) -> np.ndarray:
    """`tensor` multiplied by ``matrices[m]`` along every mode m, as
    `multi_mode_dot` does; a diagonal matrix (the identity covariates and
    their +-1 QR factors of unsupervised STACCato) scales its mode by its
    diagonal instead of a full mode product, which gives the same values
    (the products it skips are by zero)."""
    out = tensor
    for mode, matrix in enumerate(matrices):
        diagonal = np.diag(matrix)
        if matrix.shape[0] == matrix.shape[1] and np.array_equal(
            matrix, np.diag(diagonal)
        ):
            if not np.all(diagonal == 1):
                shape = [1] * out.ndim
                shape[mode] = -1
                out = out * diagonal.reshape(shape)
        else:
            out = multi_mode_dot(out, [matrix], modes=[mode])
    return out


def flip_signs(factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`factor` with each column's largest-magnitude entry made positive
    (`tensorly`'s ``svd_flip`` rule), and the signs applied."""
    rows = np.argmax(np.abs(factor), axis=0)
    signs = np.sign(factor[rows, np.arange(factor.shape[1])])
    signs[signs == 0] = 1
    return factor * signs, signs


def _coefficient_tucker(core_rot, factors_rot, Rs, core_shape, random_state):
    """The second (orthogonalizing) Tucker fit, of the coefficient tensor
    ``core_rot x_k (R_k^-1 U_k)``, without forming that tensor.

    With the thin QR ``R_k^-1 U_k = Q_k S_k``, the coefficient tensor is
    ``(core_rot x_k S_k) x_k Q_k`` with orthonormal ``Q_k``, so its Tucker
    fit at `core_shape` is the fit of the small core ``core_rot x_k S_k``
    with factors ``Q_k P_k``: what `tucker` on the full tensor returns (its
    SVD initialisation is already exact), with signs set by the same rule.
    Runs on the host whatever the active backend."""
    with tl.backend_context("numpy"):
        qs, ss = zip(
            *(
                np.linalg.qr(np.linalg.solve(R, np.asarray(U, float)))
                for R, U in zip(Rs, factors_rot, strict=True)
            )
        )
        small = multi_mode_dot(np.asarray(core_rot, float), list(ss))
        G, ps = tucker(
            small, rank=list(core_shape), init="svd", random_state=random_state
        )
        factors = []
        for mode, (q, p) in enumerate(zip(qs, ps, strict=True)):
            factor, signs = flip_signs(q @ p)
            factors.append(factor)
            shape = [1] * G.ndim
            shape[mode] = -1
            G = G * signs.reshape(shape)
    return G, factors


def staccato(
    tensor: np.ndarray,
    core_shape: Sequence[int],
    covariate: np.ndarray | None = None,
    lr_names: Sequence[str] | None = None,
    sender_names: Sequence[str] | None = None,
    receiver_names: Sequence[str] | None = None,
    random_state: int | None = None,
    init="svd",
    full_output: bool = True,
) -> StaccatoResult:
    """Re-implementation of STACCato's supervised Tucker decomposition.

    This is a from-scratch Python/TensorLy re-implementation of the algorithm
    in ``STACCato:::staccato()`` (daiqile96/STACCato, R), reconstructed by
    reading that function's decompiled body -- it is not a wrapper around the
    R package, and STACCato is not a dependency of this project. It is meant
    to let cccRISE be benchmarked against STACCato's method without an R
    runtime (this project previously ran the actual R package via an `R/`
    comparator harness, since removed in favor of this module).

    The upstream algorithm fits a Tucker decomposition whose first mode may
    be restricted to the column space of a user-supplied covariate/design
    matrix rather than a freely-estimated orthonormal factor (a supervised,
    or "envelope", Tucker model); the other modes are fixed to ``X_k = I``.
    Given per-mode covariate matrices ``X_1..X_4``, it:

    1. QR-decomposes each ``X_k = Q_k R_k``.
    2. Rotates the tensor into the ``Q``-orthonormal basis and fits an
       ordinary (unsupervised) Tucker decomposition there at ``core_shape``.
    3. Un-rotates the reconstructed tensor by ``R_k^{-1}`` to get a
       covariate-coefficient tensor.
    4. Re-decomposes that coefficient tensor with a second Tucker fit, purely
       to obtain orthonormal, directly-interpretable factor matrices and a
       core tensor (done on the first fit's core, `_coefficient_tucker`,
       which gives the same factors without forming the full tensor).
    5. Reconstructs the covariate-driven fitted tensor by mode-multiplying
       the coefficient tensor by the original (unrotated) covariate matrices.

    Only mode 1 (typically samples) accepts a supplied covariate, as upstream's
    ``staccato()`` fixes ``X_covar2..4`` to the identity: modes 2-4 (LR-pair,
    sender, receiver) are always unsupervised (``X_k = I``), so
    ``core_shape[1:]`` behaves like an ordinary Tucker rank on those modes.

    Comparing against cccRISE
    ---------------------------
    STACCato is Tucker, not CP: its four factor matrices don't share a rank
    and aren't tied together component-by-component the way cccRISE's CP
    factors (``A``/``B``/``C``/``D``, see ``ccc_rise.py``) are. Reviewer #1
    specifically asked for an ALAD-status AUC comparison, not a factor-level
    concordance score, so the two intended comparisons are:

    1. **Downstream ALAD-status AUC (the reviewer's ask).** Feed this
       function's per-sample mode-1 scores -- ``covariate @ result.W1`` when
       ``covariate`` was supplied, or ``result.W1`` directly when it wasn't
       -- into the same ANOVA/association pipeline already used to
       score cccRISE's condition factor ``A`` (see ``anova.py``), and
       report both effect sizes side by side.
    2. **Per-mode factor congruence (optional, for the concordance figure).**
       Only the LR-pair mode shares labels with cccRISE: reindex ``W2``
       (label-indexed via ``lr_names``) against cccRISE's ``D`` on shared
       LR pairs. cccRISE's ``B``/``C`` index RISE eigen-states, not cell
       types, so ``W3``/``W4`` can only be matched against pseudobulk CPD's
       ``B``/``C`` (``pseudobulk_cp_decomposition`` on the same
       ``calc_communication_score_pseudobulk`` tensor). Compute a
       cosine/Tucker congruence-coefficient matrix per mode between the two
       methods' columns, and solve the assignment with
       ``scipy.optimize.linear_sum_assignment`` to report each STACCato
       component's best-matching component. Do this per mode
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
    init : {"svd"} or (core, factors), optional
        The first Tucker fit's initialisation (``tucker``'s `init`): a given
        ``(core, factors)`` lets a caller that already has each unfolding's
        leading singular vectors skip recomputing them.
    full_output : bool
        Also form the coefficient tensor, the fitted tensor and their
        residual statistics. Without it only the factors and core are
        returned, and an unsupervised fit's `tensor` may be a tensor of the
        active `tensorly` backend (the first Tucker fit runs there; e.g. on
        the GPU).

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
    covariates = [
        covariate,
        np.eye(tensor.shape[1]),
        np.eye(tensor.shape[2]),
        np.eye(tensor.shape[3]),
    ]

    Qs: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    for X in covariates:
        Q, R = np.linalg.qr(X)
        Qs.append(Q)
        Rs.append(R)

    rotated = _mode_products(tensor, [Q.T for Q in Qs])
    core_rot, factors_rot = tucker(
        rotated, rank=list(core_shape), init=init, random_state=random_state
    )
    core_rot, factors_rot = tl.to_numpy(core_rot), list(map(tl.to_numpy, factors_rot))
    G, (W1, W2, W3, W4) = _coefficient_tucker(
        core_rot, factors_rot, Rs, core_shape, random_state
    )

    coef_tensor = fitted = None
    sigma2 = log_likelihood = float("nan")
    if full_output:
        est_rot = tucker_to_tensor((core_rot, factors_rot))
        coef_tensor = _mode_products(est_rot, [np.linalg.inv(R) for R in Rs])
        fitted = _mode_products(coef_tensor, covariates)
        residual = tensor - fitted
        sigma2 = float(np.mean(residual**2))
        log_likelihood = float(
            np.sum(-0.5 * np.log(2 * np.pi * sigma2) - residual**2 / (2 * sigma2))
        )

    def _label(
        factor: np.ndarray, names: Sequence[str] | None
    ) -> np.ndarray | pd.DataFrame:
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
