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
import pickle
from collections.abc import Callable, Sequence
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
FIT_CACHE_VERSION = 5

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


#: Non-condition modes a `ComparatorFit` keeps, in the communication tensor's
#: ``(condition, sender, receiver, lr)`` order.
OTHER_MODES = ("sender", "receiver", "lr")


@dataclass(frozen=True)
class ComparatorFit:
    """One method's factors, at the rank it was fit.

    `condition_factor` is what the ANOVA benchmark tests. The other modes
    (`modes`), and the CP `weights` or Tucker `core`, are kept for analyses
    that compare whole components across methods (NicheNet, subspace
    agreement). A Tucker `core` is in STACCato's ``(sample, lr, sender,
    receiver)`` mode order (`staccato_order`).
    """

    method: str
    rank_mode: str
    rank: str
    condition_factor: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)
    modes: dict[str, np.ndarray] = field(default_factory=dict)
    weights: np.ndarray | None = None
    core: np.ndarray | None = None

    @property
    def kind(self) -> str:
        return "tucker" if self.core is not None else "cp"


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


#: `transform_tensor` options: none; each LR pair's slice clipped at its 99th
#: percentile; each (sample, sender, receiver) block's LR vector at unit norm.
TENSOR_TRANSFORMS = ("none", "winsor", "blocknorm")


def transform_tensor(tensor: np.ndarray, kind: str) -> np.ndarray:
    """The pseudobulk tensor with large entries tempered (`TENSOR_TRANSFORMS`).

    ``winsor`` clips each LR pair at the 99th percentile of its
    (condition, sender, receiver) entries; ``blocknorm`` scales every
    (condition, sender, receiver) block's LR vector to unit norm (all-zero
    blocks stay zero), so no condition x cell-type pair dominates by magnitude.
    """
    if kind == "none":
        return tensor
    if kind == "winsor":
        cap = np.percentile(tensor.reshape(-1, tensor.shape[-1]), 99, axis=0)
        return np.minimum(tensor, cap)
    if kind == "blocknorm":
        norms = np.linalg.norm(tensor, axis=-1, keepdims=True)
        return np.divide(tensor, norms, out=np.zeros_like(tensor), where=norms > 0)
    raise ValueError(
        f"tensor transform must be one of {TENSOR_TRANSFORMS}, got {kind!r}"
    )


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


def pseudobulk_cpd_fit(
    tensor: np.ndarray, rank: int, random_state: int = 42
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Weights and ``(condition, sender, receiver, lr)`` factors of one
    pseudobulk CPD fit at `rank`."""
    weights, factors, _r2x = pseudobulk_cp_decomposition(
        tensor,
        rank,
        n_iter_max=PSEUDOBULK_N_ITER_MAX,
        tol=PSEUDOBULK_TOL,
        random_state=random_state,
    )
    return np.asarray(weights), [np.asarray(f) for f in factors]


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


def _nonneg_cp(
    tensor: np.ndarray, rank: int, random_state: int | None, backend: str = "numpy"
):
    """One Tensor-cell2cell fit, returned as NumPy arrays. ``backend="cupy"``
    runs it on the GPU (the `gpu` extra): the random initialization is drawn
    the same way, and the fit matches the CPU one to rounding."""
    if backend == "numpy":
        return non_negative_parafac(
            tl.tensor(tensor),
            rank,
            init="random",
            tol=TC2C_TOL,
            n_iter_max=TC2C_N_ITER_MAX,
            random_state=random_state,
        )
    tl.set_backend(backend)
    try:
        cp = non_negative_parafac(
            tl.tensor(tensor),
            rank,
            init="random",
            tol=TC2C_TOL,
            n_iter_max=TC2C_N_ITER_MAX,
            random_state=random_state,
        )
        weights, factors = tl.to_numpy(cp.weights), [tl.to_numpy(f) for f in cp.factors]
    finally:
        tl.set_backend("numpy")
    return tl.cp_tensor.CPTensor((weights, factors))


def _normalized_error(tensor: np.ndarray, cp) -> float:
    """``||T - T_hat|| / ||T||``, Tensor-cell2cell's elbow metric (tensorly's
    final reconstruction error when no mask is given)."""
    residual = tensor - tl.cp_to_tensor(cp)
    return float(np.sqrt(np.sum(residual**2) / np.sum(tensor**2)))


def tensor_cell2cell_elbow(
    tensor: np.ndarray,
    upper_rank: int = TC2C_UPPER_RANK,
    runs: int = TC2C_ELBOW_RUNS,
    random_state: int = 42,
    backend: str = "numpy",
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
                        tensor, _nonneg_cp(tensor, rank, random_state + run, backend)
                    )
                    for run in range(runs)
                ]
            )
            for rank in ranks
        ]
    )
    return kneedle_elbow(ranks, errors), errors


def tensor_cell2cell_fit(
    tensor: np.ndarray, rank: int, random_state: int = 42, backend: str = "numpy"
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Weights and ``(condition, sender, receiver, lr)`` factors of one
    Tensor-cell2cell fit at `rank`."""
    cp = _nonneg_cp(tensor, rank, random_state, backend)
    return np.asarray(cp.weights), [np.asarray(f) for f in cp.factors]


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


def staccato_fit(
    tensor: np.ndarray, core_shape: Sequence[int], random_state: int = 42
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Core and ``(condition, sender, receiver, lr)`` factors of an
    unsupervised STACCato fit.

    `tensor` and the returned core are in STACCato's mode order (see
    `staccato_order`); the factors are returned in the standard order.
    """
    result = staccato(tensor, core_shape, random_state=random_state)

    def _array(w: Any) -> np.ndarray:
        return w.to_numpy() if isinstance(w, pd.DataFrame) else np.asarray(w)

    factors = [_array(w) for w in (result.W1, result.W3, result.W4, result.W2)]
    return np.asarray(result.G), factors


def _comparator_fit(
    method: str,
    rank_mode: str,
    rank: str,
    factors: list[np.ndarray],
    details: dict[str, Any],
    weights: np.ndarray | None = None,
    core: np.ndarray | None = None,
) -> ComparatorFit:
    return ComparatorFit(
        method=method,
        rank_mode=rank_mode,
        rank=rank,
        condition_factor=factors[0],
        details=details,
        modes=dict(zip(OTHER_MODES, factors[1:], strict=True)),
        weights=weights,
        core=core,
    )


def _checkpointed(directory: Path | None, name: str, compute: Callable[[], Any]) -> Any:
    """`compute()`, kept in ``directory/<name>.pkl`` so that an interrupted
    run resumes after its last finished step (no checkpoint if `directory`
    is None)."""
    if directory is None:
        return compute()
    path = Path(directory) / f"{name}.pkl"
    if path.exists():
        print(f"   {name} read from checkpoint {path}")
        return pickle.loads(path.read_bytes())
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_bytes(pickle.dumps(result))
    partial.replace(path)
    return result


def fit_comparators(
    tensor: np.ndarray,
    rank_mode: str,
    shared_rank: int,
    tc2c_upper_rank: int = TC2C_UPPER_RANK,
    tc2c_elbow_runs: int = TC2C_ELBOW_RUNS,
    staccato_varexp: float = STACCATO_VAREXP,
    random_state: int = 42,
    tc2c_backend: str = "numpy",
    pb_max_rank: int | None = None,
    checkpoint_dir: Path | None = None,
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
    checkpoint_dir : Path, optional
        Where the rank searches and the fits are checkpointed (one file each);
        it must be specific to the tensor and the settings.
    """
    if rank_mode not in RANK_MODES:
        raise ValueError(f"rank_mode must be one of {RANK_MODES}, got {rank_mode!r}")

    stac_tensor = staccato_order(tensor)
    if rank_mode == "own":
        pb_rank, diagnostics = _checkpointed(
            checkpoint_dir,
            "pseudobulk_cpd_rank",
            lambda: pseudobulk_cpd_rank(
                tensor, max_rank=pb_max_rank, random_state=random_state
            ),
        )
        pb_details: dict[str, Any] = {
            "corcondia_threshold": CORCONDIA_THRESHOLD,
            "core_consistency": diagnostics["CoreConsistency"].tolist(),
            "max_modes_collinear": diagnostics["MaxModesCollinear"].tolist(),
        }
        tc2c_rank, errors = _checkpointed(
            checkpoint_dir,
            "tensor_cell2cell_elbow",
            lambda: tensor_cell2cell_elbow(
                tensor,
                upper_rank=tc2c_upper_rank,
                runs=tc2c_elbow_runs,
                random_state=random_state,
                backend=tc2c_backend,
            ),
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

    pb_weights, pb_factors = _checkpointed(
        checkpoint_dir,
        f"pseudobulk_cpd_fit_{pb_rank}",
        lambda: pseudobulk_cpd_fit(tensor, pb_rank, random_state),
    )
    tc_weights, tc_factors = _checkpointed(
        checkpoint_dir,
        f"tensor_cell2cell_fit_{tc2c_rank}",
        lambda: tensor_cell2cell_fit(tensor, tc2c_rank, random_state, tc2c_backend),
    )
    core, st_factors = staccato_fit(stac_tensor, core_shape, random_state)
    return [
        _comparator_fit(
            "Pseudobulk CPD",
            rank_mode,
            str(pb_rank),
            pb_factors,
            pb_details,
            pb_weights,
        ),
        _comparator_fit(
            "Tensor-cell2cell",
            rank_mode,
            str(tc2c_rank),
            tc_factors,
            tc2c_details,
            tc_weights,
        ),
        _comparator_fit(
            "STACCato",
            rank_mode,
            "x".join(str(r) for r in core_shape),
            st_factors,
            stac_details | {"core_shape": list(core_shape)},
            core=core,
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
    """Cache fits' raw (not read-depth corrected) factors: the condition factor
    as ``factor_<i>``, the other modes as ``<mode>_<i>``, and the CP weights or
    Tucker core as ``weights_<i>`` / ``core_<i>``.

    `key` is everything the fits depend on; `load_fits` refuses the cache when
    it differs. `tensor_info` describes the tensor they were fit on, so a run
    served from the cache can still report it without rebuilding it.
    """
    data_vars: dict[str, xr.DataArray] = {}
    for i, fit in enumerate(fits):
        data_vars[f"factor_{i}"] = xr.DataArray(
            np.asarray(fit.condition_factor, dtype=float),
            dims=("condition", f"component_{i}"),
            attrs={
                "method": fit.method,
                "rank_mode": fit.rank_mode,
                "rank": fit.rank,
                "details": json.dumps(fit.details),
            },
        )
        data_vars |= _fit_arrays(i, fit)
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
    return _read_all(ds)


def read_fits(path: Path) -> tuple[list[ComparatorFit], dict[str, Any], np.ndarray]:
    """Every cached fit, its tensor info and its condition labels, without the
    cache-key check: for analyses that consume the runner's fits."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Produce it with\n    uv run cccrise-benchmark --dataset <dataset>"
        )
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    fits, info = _read_all(ds)
    return fits, info, ds.coords["condition"].values.astype(str)


def _read_all(ds: xr.Dataset) -> tuple[list[ComparatorFit], dict[str, Any]]:
    n_fits = sum(1 for name in ds.data_vars if str(name).startswith("factor_"))
    fits = [_read_fit(ds, i) for i in range(n_fits)]
    return fits, json.loads(ds.attrs["tensor_info"])


def _fit_arrays(i: int, fit: ComparatorFit) -> dict[str, xr.DataArray]:
    """The non-condition arrays of fit `i`, for `save_fits`."""
    out = {
        f"{mode}_{i}": xr.DataArray(
            np.asarray(values, dtype=float),
            dims=(f"{mode}_row_{i}", f"{mode}_component_{i}"),
        )
        for mode, values in fit.modes.items()
    }
    if fit.weights is not None:
        out[f"weights_{i}"] = xr.DataArray(
            np.asarray(fit.weights, dtype=float), dims=(f"component_{i}",)
        )
    if fit.core is not None:
        out[f"core_{i}"] = xr.DataArray(
            np.asarray(fit.core, dtype=float),
            dims=[f"core_{i}_{k}" for k in range(np.ndim(fit.core))],
        )
    return out


def _read_fit(ds: xr.Dataset, i: int) -> ComparatorFit:
    var = ds[f"factor_{i}"]
    return ComparatorFit(
        method=var.attrs["method"],
        rank_mode=var.attrs["rank_mode"],
        rank=var.attrs["rank"],
        condition_factor=var.values,
        details=json.loads(var.attrs["details"]),
        modes={m: ds[f"{m}_{i}"].values for m in OTHER_MODES if f"{m}_{i}" in ds},
        weights=ds[f"weights_{i}"].values if f"weights_{i}" in ds else None,
        core=ds[f"core_{i}"].values if f"core_{i}" in ds else None,
    )
