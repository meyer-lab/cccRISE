"""Tensor-method comparators for the cccRISE ANOVA benchmark.

Every method here reduces to one thing: a ``(n_conditions, n_components)``
condition factor, rows in ``np.unique(obs[condition_column])`` order. That
matrix goes through the same `anova.run_multiple_factors_anova` call cccRISE's
own condition factor does, so the only thing that differs between methods is
the decomposition.

Both comparators decompose the same cell-type pseudobulk tensor -- mean
expression per (condition, cell type), scored with the Tensor-cell2cell
expression-product builder cccRISE already uses (`ccc_rise
.calc_communication_score_pseudobulk`):

- **Tensor-cell2cell**: non-negative CP, with Tensor-cell2cell's own
  ``tf_optimization='regular'`` settings and its elbow rank selection. The
  tensor is the shared one, not upstream's: product rather than mean scoring,
  cell types intersected rather than unioned, and an absent cell type
  zero-filled rather than masked.
- **STACCato**: the `comparators.staccato` Tucker re-implementation, run
  *unsupervised* (no mode-1 covariate). Supplying the phenotype as STACCato's
  covariate would put the labels into the sample scores, and an ANOVA of those
  scores against the same labels would be circular.

Each comparator's rank is either selected by that method's own published
procedure (`rank_mode="own"`) or set to a single rank shared by every method
(`rank_mode="shared"`). cccRISE is never refit: its ranks were selected by
BiCV and CORCONDIA and its published factors are read from the cached model.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anndata
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_parafac

from .ccc_rise import calc_communication_score_pseudobulk
from .comparators.staccato import staccato
from .utils import pseudobulk_X

RANK_MODES = ("own", "shared")

# Tensor-cell2cell's `run_tensor_cell2cell_pipeline` defaults with
# tf_optimization='regular': random init, 10 elbow runs per rank up to rank 25,
# tol 1e-7, 100 iterations.
TC2C_UPPER_RANK = 25
TC2C_ELBOW_RUNS = 10
TC2C_TOL = 1e-7
TC2C_N_ITER_MAX = 100

# STACCato's tutorial keeps every component above 1% explained variance per mode
# (`cal_eigen_varexp(..., varexp = 1)`).
STACCATO_VAREXP = 1.0


@dataclass(frozen=True)
class ComparatorFit:
    """One method's condition factor, at the rank it was fit."""

    method: str
    rank_mode: str
    rank: str
    condition_factor: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)


def pseudobulk_tensor(
    adata: anndata.AnnData,
    condition_column: str,
    celltype_column: str,
    lr_pairs: pd.DataFrame,
    complex_sep: str | None = "&",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Cell-type pseudobulk communication tensor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        The tensor, shape ``(n_conditions, n_celltypes, n_celltypes,
        n_lr_pairs)`` with conditions in `np.unique` order; the cell-type
        names of the sender/receiver modes; and the LR pairs of the last mode.
    """
    matrices = pseudobulk_X(
        adata, condition_name=condition_column, groupby=celltype_column, type="mean"
    )
    tensor, filtered_lr_pairs = calc_communication_score_pseudobulk(
        matrices, lr_pairs=lr_pairs, complex_sep=complex_sep
    )
    celltypes = np.unique(adata.obs[celltype_column].unique())
    return tensor, celltypes, filtered_lr_pairs


def kneedle_elbow(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, S: float = 1.0
) -> int:
    """Elbow of a convex, decreasing curve by offline Kneedle.

    Port of what Tensor-cell2cell calls (`kneed.KneeLocator(curve="convex",
    direction="decreasing", S=1.0).elbow`) without the interpolation step,
    which is a no-op on integer ranks. Returns the first local maximum of the
    difference curve that the curve then drops below its threshold from, or
    the difference curve's argmax if none does.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 3:
        return int(xs[np.argmin(ys)])

    def _normalize(v: np.ndarray) -> np.ndarray:
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else np.zeros_like(v)

    x_norm = _normalize(xs)
    y_norm = _normalize(ys)
    difference = (y_norm.max() - y_norm) - x_norm

    maxima = [
        i
        for i in range(1, xs.size - 1)
        if difference[i] >= difference[i - 1] and difference[i] >= difference[i + 1]
    ]
    step = S * float(np.abs(np.diff(x_norm)).mean())
    for i in maxima:
        threshold = difference[i] - step
        later_maxima = [j for j in maxima if j > i]
        stop = later_maxima[0] if later_maxima else xs.size
        if np.any(difference[i + 1 : stop] < threshold):
            return int(xs[i])
    return int(xs[int(np.argmax(difference))])


def _nonneg_cp(tensor: np.ndarray, rank: int, random_state: int | None):
    return non_negative_parafac(
        tl.tensor(tensor),
        rank,
        init="random",
        tol=TC2C_TOL,
        n_iter_max=TC2C_N_ITER_MAX,
        random_state=random_state,
    )


def _normalized_error(tensor: np.ndarray, cp) -> float:
    """``||T - T_hat||^2 / ||T||^2``, Tensor-cell2cell's elbow metric."""
    residual = tensor - tl.cp_to_tensor(cp)
    return float(np.sum(residual**2) / np.sum(tensor**2))


def tensor_cell2cell_elbow(
    tensor: np.ndarray,
    upper_rank: int = TC2C_UPPER_RANK,
    runs: int = TC2C_ELBOW_RUNS,
    random_state: int = 42,
) -> tuple[int, np.ndarray]:
    """Tensor-cell2cell's rank: the elbow of the mean normalized error across
    `runs` random-init fits at every rank ``1..upper_rank``.

    Returns the selected rank and the mean error per rank.
    """
    ranks = list(range(1, upper_rank + 1))
    errors = np.array(
        [
            np.mean(
                [
                    _normalized_error(
                        tensor, _nonneg_cp(tensor, rank, random_state + run)
                    )
                    for run in range(runs)
                ]
            )
            for rank in ranks
        ]
    )
    return kneedle_elbow(ranks, errors), errors


def tensor_cell2cell_factor(
    tensor: np.ndarray, rank: int, random_state: int = 42
) -> np.ndarray:
    """Condition factor of one Tensor-cell2cell fit at `rank`."""
    cp = _nonneg_cp(tensor, rank, random_state)
    return np.asarray(cp.factors[0])


def eigen_varexp_rank(
    tensor: np.ndarray, mode: int, varexp: float = STACCATO_VAREXP
) -> tuple[int, np.ndarray]:
    """STACCato's per-mode rank (`cal_eigen_varexp`): the number of
    eigenvalues of the mode-`mode` unfolding's Gram matrix explaining more
    than `varexp` percent of its variance.

    Returns the rank and the explained-variance proportions.
    """
    unfolded = tl.unfold(tl.tensor(tensor), mode)
    eigenvalues = np.linalg.eigvalsh(unfolded @ unfolded.T)[::-1]
    proportions = np.clip(eigenvalues, 0.0, None) / eigenvalues.sum()
    return max(int(np.sum(proportions > varexp / 100)), 1), proportions


def staccato_order(tensor: np.ndarray) -> np.ndarray:
    """``(condition, sender, receiver, lr)`` -> STACCato's ``(sample, lr,
    sender, receiver)``."""
    return np.transpose(tensor, (0, 3, 1, 2))


def staccato_factor(
    tensor: np.ndarray, core_shape: Sequence[int], random_state: int = 42
) -> np.ndarray:
    """Sample-mode factor ``W1`` of an unsupervised STACCato fit.

    `tensor` is in STACCato's mode order (see `staccato_order`).
    """
    return np.asarray(staccato(tensor, core_shape, random_state=random_state).W1)


def fit_comparators(
    tensor: np.ndarray,
    rank_mode: str,
    shared_rank: int,
    tc2c_upper_rank: int = TC2C_UPPER_RANK,
    tc2c_elbow_runs: int = TC2C_ELBOW_RUNS,
    staccato_varexp: float = STACCATO_VAREXP,
    random_state: int = 42,
) -> list[ComparatorFit]:
    """Fit every comparator on the pseudobulk `tensor` under one rank mode.

    Parameters
    ----------
    tensor : np.ndarray
        ``(n_conditions, n_celltypes, n_celltypes, n_lr_pairs)``, from
        `pseudobulk_tensor`.
    rank_mode : {"own", "shared"}
        ``"own"`` lets each method select its rank by its own procedure;
        ``"shared"`` fits every method at `shared_rank` (STACCato at
        `shared_rank` in every mode, capped at the mode's size).
    shared_rank : int
        Rank used when `rank_mode="shared"`.
    """
    if rank_mode not in RANK_MODES:
        raise ValueError(f"rank_mode must be one of {RANK_MODES}, got {rank_mode!r}")

    stac_tensor = staccato_order(tensor)
    if rank_mode == "own":
        tc2c_rank, errors = tensor_cell2cell_elbow(
            tensor,
            upper_rank=tc2c_upper_rank,
            runs=tc2c_elbow_runs,
            random_state=random_state,
        )
        tc2c_details: dict[str, Any] = {"elbow_errors": errors.tolist()}
        selections = [
            eigen_varexp_rank(stac_tensor, mode, staccato_varexp) for mode in range(4)
        ]
        core_shape = tuple(rank for rank, _ in selections)
        stac_details: dict[str, Any] = {
            "varexp_percent": staccato_varexp,
            "top_proportions": [p[:10].tolist() for _, p in selections],
        }
    else:
        tc2c_rank = shared_rank
        tc2c_details = {}
        core_shape = tuple(min(shared_rank, size) for size in stac_tensor.shape)
        stac_details = {}

    return [
        ComparatorFit(
            method="Tensor-cell2cell",
            rank_mode=rank_mode,
            rank=str(tc2c_rank),
            condition_factor=tensor_cell2cell_factor(tensor, tc2c_rank, random_state),
            details=tc2c_details,
        ),
        ComparatorFit(
            method="STACCato",
            rank_mode=rank_mode,
            rank="x".join(str(r) for r in core_shape),
            condition_factor=staccato_factor(stac_tensor, core_shape, random_state),
            details=stac_details | {"core_shape": list(core_shape)},
        ),
    ]
