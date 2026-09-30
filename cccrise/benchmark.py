"""Tensor-method comparators for the cccRISE ANOVA benchmark.

Every method here reduces to one thing: a ``(n_conditions, n_components)``
condition factor, rows in ``np.unique(obs[condition_column])`` order. That
matrix goes through the same `anova.run_multiple_factors_anova` call cccRISE's
own condition factor does, so the only thing that differs between methods is
the decomposition.

Every comparator decomposes the same cell-type pseudobulk tensor -- mean
expression per (condition, cell type), scored with the Tensor-cell2cell
expression-product builder cccRISE already uses (`ccc_rise
.calc_communication_score_pseudobulk`):

- **Pseudobulk CPD**: the manuscript's own baseline, signed CP exactly as
  `ccc_rise.pseudobulk_cp_decomposition` fits it, with its rank selected by
  the criteria cccRISE's CP rank was (CORCONDIA and the collinearity screen).
- **Tensor-cell2cell**: non-negative CP, with Tensor-cell2cell's own
  ``tf_optimization='regular'`` settings and its elbow rank selection. The
  tensor is the shared one, not upstream's: product rather than mean scoring,
  and every cell type kept, with a condition's absent cell type zero-filled
  rather than masked.
- **STACCato**: the `comparators.staccato` Tucker re-implementation, run
  *unsupervised* (no mode-1 covariate). Supplying the phenotype as STACCato's
  covariate would put the labels into the sample scores, and an ANOVA of those
  scores against the same labels would be circular.

Each comparator's rank is either selected by that method's own published
procedure (`rank_mode="own"`) or set to a single rank shared by every method
(`rank_mode="shared"`). cccRISE is never refit: its ranks were selected by
BiCV and CORCONDIA and its published factors are read from the cached model.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import tensorly as tl
import xarray as xr
from tensorly.decomposition import non_negative_parafac

from .ccc_rise import calc_communication_score_pseudobulk, pseudobulk_cp_decomposition
from .comparators.staccato import staccato
from .rank_selection import cp_core_consistency_sweep, cpd_component_collinearity
from .utils import pseudobulk_X

RANK_MODES = ("own", "shared")

# Bump when a change to a comparator's fitting or rank selection should
# invalidate every cached own-rank fit.
FIT_CACHE_VERSION = 2

# `tensor.run_pseudobulk_workflow`'s CP settings, and the rule cccRISE's CP rank
# was read off by: the last rank before core consistency first drops below 85%
# or a collinear component pair appears.
PSEUDOBULK_N_ITER_MAX = 1000
PSEUDOBULK_TOL = 1e-11
CORCONDIA_THRESHOLD = 85.0

# Tensor-cell2cell's `run_tensor_cell2cell_pipeline` defaults with
# tf_optimization='regular': random init, 10 elbow runs per rank up to rank 25,
# tol 1e-7, 100 iterations.
TC2C_UPPER_RANK = 25
TC2C_ELBOW_RUNS = 10
TC2C_TOL = 1e-7
TC2C_N_ITER_MAX = 100

# STACCato's tutorial keeps every LR-pair, sender and receiver component above
# 1% explained variance (`cal_eigen_varexp(..., varexp = 1)`); run
# unsupervised, the sample mode is cut the same way.
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


def pseudobulk_cpd_rank(
    tensor: np.ndarray,
    max_rank: int | None = None,
    threshold: float = CORCONDIA_THRESHOLD,
    random_state: int = 42,
) -> tuple[int, pd.DataFrame]:
    """Pseudobulk CPD's rank, by the rule cccRISE's CP rank was selected with.

    Fits ranks ``1..max_rank`` (default: the tensor's smallest mode, the bound
    `ccc_rise.validate_cp_rank` puts on cccRISE) and returns the last rank
    before core consistency first drops below `threshold` percent or any
    component pair turns collinear, with the per-rank diagnostics.
    """
    max_rank = max_rank or min(tensor.shape)
    factors_by_rank = {
        rank: pseudobulk_cp_decomposition(
            tensor,
            rank,
            n_iter_max=PSEUDOBULK_N_ITER_MAX,
            tol=PSEUDOBULK_TOL,
            random_state=random_state,
        )[:2]
        for rank in range(1, max_rank + 1)
    }
    diagnostics = cp_core_consistency_sweep(tensor, factors_by_rank).merge(
        cpd_component_collinearity(factors_by_rank), on="Rank"
    )
    passes = (diagnostics["CoreConsistency"] >= threshold) & (
        diagnostics["MaxModesCollinear"] == 0
    )
    first_fail = np.flatnonzero(~passes.to_numpy())
    rank = int(first_fail[0]) if first_fail.size else max_rank
    return max(rank, 1), diagnostics


def pseudobulk_cpd_factor(
    tensor: np.ndarray, rank: int, random_state: int = 42
) -> np.ndarray:
    """Condition factor of one pseudobulk CPD fit at `rank`."""
    _weights, factors, _r2x = pseudobulk_cp_decomposition(
        tensor,
        rank,
        n_iter_max=PSEUDOBULK_N_ITER_MAX,
        tol=PSEUDOBULK_TOL,
        random_state=random_state,
    )
    return np.asarray(factors[0])


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
    """``||T - T_hat||^2 / ||T||^2``, the square of Tensor-cell2cell's elbow
    metric ``||T - T_hat|| / ||T||``; squaring can move the elbow."""
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
        pb_rank, diagnostics = pseudobulk_cpd_rank(tensor, random_state=random_state)
        pb_details: dict[str, Any] = {
            "corcondia_threshold": CORCONDIA_THRESHOLD,
            "core_consistency": diagnostics["CoreConsistency"].tolist(),
            "max_modes_collinear": diagnostics["MaxModesCollinear"].tolist(),
        }
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
        pb_rank = shared_rank
        pb_details = {}
        tc2c_rank = shared_rank
        tc2c_details = {}
        core_shape = tuple(min(shared_rank, size) for size in stac_tensor.shape)
        stac_details = {}

    return [
        ComparatorFit(
            method="Pseudobulk CPD",
            rank_mode=rank_mode,
            rank=str(pb_rank),
            condition_factor=pseudobulk_cpd_factor(tensor, pb_rank, random_state),
            details=pb_details,
        ),
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


def fit_settings() -> dict[str, Any]:
    """Every module-level constant an own-rank fit depends on, for the cache key."""
    return {
        "fit_cache_version": FIT_CACHE_VERSION,
        "pseudobulk_n_iter_max": PSEUDOBULK_N_ITER_MAX,
        "pseudobulk_tol": PSEUDOBULK_TOL,
        "corcondia_threshold": CORCONDIA_THRESHOLD,
        "tc2c_tol": TC2C_TOL,
        "tc2c_n_iter_max": TC2C_N_ITER_MAX,
    }


def save_fits(
    path: Path,
    fits: Sequence[ComparatorFit],
    conditions: np.ndarray,
    key: dict[str, Any],
    tensor_info: dict[str, Any],
) -> None:
    """Cache fits' raw (not read-depth corrected) condition factors.

    `key` is everything the fits depend on; `load_fits` refuses the cache when
    it differs. `tensor_info` describes the tensor they were fit on, so a run
    served from the cache can still report it without rebuilding it.
    """
    data_vars = {
        f"factor_{i}": xr.DataArray(
            np.asarray(fit.condition_factor, dtype=float),
            dims=("condition", f"component_{i}"),
            attrs={
                "method": fit.method,
                "rank_mode": fit.rank_mode,
                "rank": fit.rank,
                "details": json.dumps(fit.details),
            },
        )
        for i, fit in enumerate(fits)
    }
    ds = xr.Dataset(
        data_vars,
        coords={"condition": np.asarray(conditions).astype(str)},
        attrs={
            "cache_key": json.dumps(key, sort_keys=True),
            "tensor_info": json.dumps(tensor_info),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def load_fits(
    path: Path, conditions: np.ndarray, key: dict[str, Any]
) -> tuple[list[ComparatorFit], dict[str, Any]] | None:
    """Cached fits and their tensor info, or None when the cache is missing or
    was made from different data or settings."""
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    if ds.attrs.get("cache_key") != json.dumps(key, sort_keys=True):
        print(f"   {path.name} was made with different settings; refitting")
        return None
    if not np.array_equal(
        ds.coords["condition"].values, np.asarray(conditions).astype(str)
    ):
        print(f"   {path.name} was fit on different conditions; refitting")
        return None
    fits = [
        ComparatorFit(
            method=var.attrs["method"],
            rank_mode=var.attrs["rank_mode"],
            rank=var.attrs["rank"],
            condition_factor=var.values,
            details=json.loads(var.attrs["details"]),
        )
        for var in (ds[f"factor_{i}"] for i in range(len(ds.data_vars)))
    ]
    return fits, json.loads(ds.attrs["tensor_info"])
