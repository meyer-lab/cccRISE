"""Each annotation-pooled comparator's own rank selection and fit.

- **Pseudobulk CPD** is fit exactly as cccRISE's CP (`ccc_rise.CP_FIT_OPTIONS`
  through `rank_selection.fit_cp_on_device`), and its rank sweep is cccRISE's
  (`runners.run_cpd_rank.start_sweep`), so the two differ only in what they
  decompose.
- **Tensor-cell2cell**: non-negative CP, its rank the Kneedle elbow of the
  normalized error averaged over `TC2C_RUNS` random starts per rank
  1..`TC2C_UPPER_RANKS` (by dataset and labelling), as its package's
  ``run_tensor_cell2cell_pipeline``. On the masked tensor the package forces
  random starts, and the fit is the lowest-error start at the elbow rank (its
  "robust" preset keeps the best of many runs; `best_start`). On the paper's
  unmasked ``inner`` tensor the fit is one SVD start, as the paper's BALF
  analysis (`tensor_cell2cell_fit`'s `init`). Deviations meant to let it
  perform its best: fits run with cccRISE's tolerance and precision for up to
  `TC2C_N_ITER_MAX` iterations (upstream 100 at tol 1e-6), the rank is
  Kneedle's (the paper chose by eye), and the best start is kept; the
  iteration and rank caps keep the sweep's runtime within reach on IBD.
  The tensor's missing entries are masked in the fit and left out of the
  error (`masked_error`), as in Tensor-cell2cell.
- **STACCato**, on its own tensor (`tensors.staccato_tensor`) and
  unsupervised (no sample covariate): each mode keeps the
  components whose eigenvalue share of the mode's unfolding is above
  `STACCATO_VAREXP` percent (``cal_eigen_varexp``), and the Tucker fit is
  `comparators.staccato.staccato`.

Seeds start at 42 throughout (upstream Tensor-cell2cell is unseeded).
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
import tensorly as tl
import xarray as xr
from tensorly.decomposition import non_negative_parafac

from ..rank_selection import (
    CP_DTYPE,
    Factors,
    cp_backend,
    kneedle_elbow,
    max_congruence,
)
from .staccato import staccato
from .tensors import PooledTensor

#: Top of pseudobulk CPD's rank sweep per dataset and labelling (each also
#: stops at the tensor's smallest mode): fine labels' larger tensor is capped
#: lower to bound its runtime on IBD.
PB_UPPER_RANKS = {
    "alad": {"coarse": 25, "fine": 25},
    "ibd": {"coarse": 25, "fine": 20},
}
#: Top of the elbow sweep per dataset and labelling: fine labels' larger tensor
#: is capped lower to bound its runtime, and IBD's masked fits are memory hungry.
TC2C_UPPER_RANKS = {
    "alad": {"coarse": 25, "fine": 20},
    "ibd": {"coarse": 15, "fine": 15},
}
TC2C_N_ITER_MAX = 1000
TC2C_RUNS = 10
STACCATO_VAREXP = 1.0
#: The communication tensor's modes, in order.
MODES = ("sample", "sender", "receiver", "lr")
#: Tensor-cell2cell's and STACCato's mode order.
TC2C_ORDER = STACCATO_ORDER = ("sample", "lr", "sender", "receiver")


def masked_error(
    tensor: np.ndarray, mask: np.ndarray | None, weights, factors
) -> float:
    """``||M (T - T_hat)|| / ||M T||``, Tensor-cell2cell's normalized error,
    over the observed entries only (all of them when `mask` is None)."""
    residual = tensor - tl.cp_to_tensor((weights, factors))
    if mask is not None:
        residual, tensor = residual * mask, tensor * mask
    return float(np.linalg.norm(residual) / np.linalg.norm(tensor))


def tensor_cell2cell_fit(
    tensor: np.ndarray,
    mask: np.ndarray | None,
    rank: int,
    seed: int,
    n_iter_max: int,
    tol: float,
    device: str | None = None,
    dtype: str = CP_DTYPE,
    init: str = "random",
) -> tuple[np.ndarray, Factors]:
    """One Tensor-cell2cell fit from the `init` start (``"random"``, seeded
    by `seed`, or ``"svd"``, the package's truncated SVD), in `dtype` on
    `device` (`rank_selection.cp_backend`): NumPy weights and factors.

    It is fit in Tensor-cell2cell's mode order (`TC2C_ORDER`), so the random
    start of a seed, and with it the fit, is upstream's."""
    to_upstream = [MODES.index(m) for m in TC2C_ORDER]
    with tl.backend_context(cp_backend(device)):
        cp = non_negative_parafac(
            tl.tensor(np.transpose(tensor, to_upstream).astype(dtype)),
            rank,
            init=init,
            svd="truncated_svd",
            mask=None
            if mask is None
            else tl.tensor(np.transpose(mask, to_upstream).astype(dtype)),
            tol=tol,
            n_iter_max=n_iter_max,
            random_state=seed,
        )
        factors = dict(zip(TC2C_ORDER, cp.factors, strict=True))
        weights = tl.to_numpy(cp.weights).astype(np.float64)
        return weights, [tl.to_numpy(factors[m]).astype(np.float64) for m in MODES]


#: Each elbow worker process's tensor, mask and fit settings
#: (`_init_elbow_worker`), sent once rather than with every fit.
_WORKER: dict = {}


def _init_elbow_worker(tensor, mask, fit_kwargs) -> None:
    _WORKER.update(tensor=tensor, mask=mask, fit_kwargs=fit_kwargs)


def _elbow_fit(rank: int, seed: int) -> tuple[float, Factors]:
    """One elbow fit and its error, on the worker's tensor."""
    tensor, mask = _WORKER["tensor"], _WORKER["mask"]
    weights, factors = tensor_cell2cell_fit(
        tensor, mask, rank, seed, **_WORKER["fit_kwargs"]
    )
    return masked_error(tensor, mask, weights, factors), factors


def tensor_cell2cell_elbow(
    tensor: np.ndarray,
    mask: np.ndarray | None,
    ranks,
    runs: int = TC2C_RUNS,
    random_state: int = 42,
    workers: int = 1,
    **fit_kwargs,
) -> tuple[pd.DataFrame, dict[int, dict[int, Factors]]]:
    """`runs` fits (seeds `random_state` + run; `fit_kwargs` are
    `tensor_cell2cell_fit`'s) at every rank, spread over `workers` processes:
    one row per fit ('Rank', 'Run', 'Seed', its normalized 'Error' and
    'FitR2X' over the observed entries), and each rank's factors by seed.
    The fits are independent and seeded, so `workers` changes only how fast
    they run; it pays on small tensors, where a fit is bound by per-step
    overhead rather than arithmetic and a GPU sits idle."""
    tasks = [(rank, random_state + run) for rank in ranks for run in range(runs)]
    args = (tensor, mask, fit_kwargs)
    if workers > 1:
        pool = ProcessPoolExecutor(
            workers,
            mp_context=get_context("spawn"),
            initializer=_init_elbow_worker,
            initargs=args,
        )
        results = pool.map(_elbow_fit, *zip(*tasks, strict=True))
    else:
        pool = None
        _init_elbow_worker(*args)
        results = (_elbow_fit(*task) for task in tasks)
    rows, factors_by_rank = [], {}
    for (rank, seed), (error, factors) in zip(tasks, results, strict=True):
        factors_by_rank.setdefault(rank, {})[seed] = factors
        rows.append(
            {"Rank": rank, "Run": seed - random_state, "Seed": seed, "Error": error}
            | {"FitR2X": 1 - error**2}
        )
        if len(factors_by_rank[rank]) == runs:
            mean = np.mean([r["Error"] for r in rows[-runs:]])
            print(f"   rank {rank}: mean error {mean:.4f}", flush=True)
    if pool is not None:
        pool.shutdown()
    return pd.DataFrame(rows), factors_by_rank


def best_start(
    elbow: pd.DataFrame, factors_by_rank: dict[int, dict[int, Factors]], rank: int
) -> tuple[int, np.ndarray, Factors]:
    """The lowest-error start of `tensor_cell2cell_elbow` at `rank`: its seed,
    weights (ones: the fits' factors carry their scale) and factors."""
    at_rank = elbow[elbow["Rank"] == rank]
    seed = int(at_rank.loc[at_rank["Error"].idxmin(), "Seed"])
    return seed, np.ones(rank), factors_by_rank[rank][seed]


def starts_dataset(
    pooled: PooledTensor, factors_by_rank: dict[int, dict[int, Factors]]
) -> xr.Dataset:
    """Every start's factors (`tensor_cell2cell_elbow`'s), labelled: per mode
    of `MODES`, ``<mode>_factor`` with one column per component of every
    start along ``start_component``, whose 'rank', 'seed' and 'component'
    coordinates say which."""
    columns = [
        (rank, seed, component)
        for rank, by_seed in factors_by_rank.items()
        for seed in by_seed
        for component in range(1, rank + 1)
    ]
    stacked = [
        np.hstack(
            [f[mode] for by_seed in factors_by_rank.values() for f in by_seed.values()]
        )
        for mode in range(len(MODES))
    ]
    rank, seed, component = (np.array(c) for c in zip(*columns, strict=True))
    return xr.Dataset(
        {
            f"{m}_factor": ((m, "start_component"), f)
            for m, f in zip(MODES, stacked, strict=True)
        },
        coords=_mode_labels(pooled)
        | {
            "rank": ("start_component", rank),
            "seed": ("start_component", seed),
            "component": ("start_component", component),
        },
    )


def starts_factors(starts: xr.Dataset) -> dict[int, dict[int, Factors]]:
    """`starts_dataset`'s factors back as each rank's factors by seed."""
    ranks, seeds = starts["rank"].to_numpy(), starts["seed"].to_numpy()
    factors_by_rank: dict[int, dict[int, Factors]] = {}
    for rank, seed in sorted(set(zip(ranks.tolist(), seeds.tolist(), strict=True))):
        columns = np.flatnonzero((ranks == rank) & (seeds == seed))
        factors_by_rank.setdefault(rank, {})[seed] = [
            starts[f"{m}_factor"].to_numpy()[:, columns] for m in MODES
        ]
    return factors_by_rank


def start_congruence(starts: xr.Dataset) -> pd.DataFrame:
    """Each start's `rank_selection.max_congruence`, from `starts_dataset`'s
    factors: one row per start ('Rank', 'Seed', 'MaxCongruence')."""
    ranks, seeds = starts["rank"].to_numpy(), starts["seed"].to_numpy()
    rows = []
    for rank, seed in sorted(set(zip(ranks.tolist(), seeds.tolist(), strict=True))):
        columns = np.flatnonzero((ranks == rank) & (seeds == seed))
        factors = [starts[f"{m}_factor"].to_numpy()[:, columns] for m in MODES]
        rows.append(
            {"Rank": rank, "Seed": seed, "MaxCongruence": max_congruence(factors)}
        )
    return pd.DataFrame(rows)


def tensor_cell2cell_rank(elbow: pd.DataFrame) -> int | None:
    """Tensor-cell2cell's rank: the Kneedle elbow of the mean error by rank."""
    mean = elbow.groupby("Rank")["Error"].mean()
    return kneedle_elbow(mean.index, mean.to_numpy())


def eigenvalue_shares(tensor: np.ndarray) -> pd.DataFrame:
    """Each mode's eigenvalue shares of its unfolding's Gram matrix, largest
    first: one row per ('Mode', 'Component', 'Share'), modes in `MODES`
    order."""
    frames = []
    for mode, name in enumerate(MODES):
        unfolded = tl.unfold(tensor, mode)
        values = np.clip(np.linalg.eigvalsh(unfolded @ unfolded.T)[::-1], 0, None)
        frames.append(
            pd.DataFrame(
                {
                    "Mode": name,
                    "Component": np.arange(1, values.size + 1),
                    "Share": values / values.sum(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def staccato_core_shape(
    shares: pd.DataFrame, varexp: float = STACCATO_VAREXP
) -> dict[str, int]:
    """STACCato's rank per mode: the components above `varexp` percent."""
    above = shares[shares["Share"] > varexp / 100].groupby("Mode").size()
    return {mode: max(int(above.get(mode, 0)), 1) for mode in MODES}


def staccato_fit(
    tensor: np.ndarray, core_shape: dict[str, int], random_state: int = 42
) -> tuple[np.ndarray, Factors]:
    """An unsupervised STACCato fit: the core, in `MODES` order, and the
    factors of `MODES`."""
    to_stac = [MODES.index(m) for m in STACCATO_ORDER]
    result = staccato(
        np.transpose(tensor, to_stac),
        [core_shape[m] for m in STACCATO_ORDER],
        random_state=random_state,
    )
    factors = dict(
        zip(STACCATO_ORDER, (result.W1, result.W2, result.W3, result.W4), strict=True)
    )
    core = np.transpose(result.G, [STACCATO_ORDER.index(m) for m in MODES])
    return core, [np.asarray(factors[m]) for m in MODES]


def fit_dataset(
    pooled: PooledTensor,
    factors: Factors,
    weights: np.ndarray | None = None,
    core: np.ndarray | None = None,
    **attrs,
) -> xr.Dataset:
    """A fit with every mode labelled: ``<mode>_factor`` per mode of `MODES`
    and the CP ``weights`` (components along ``component``) or Tucker
    ``core`` (each mode's components along ``<mode>_component``)."""
    labels = _mode_labels(pooled)
    rank = "component" if core is None else None
    data: dict[str, tuple] = {
        f"{m}_factor": ((m, rank or f"{m}_component"), f)
        for m, f in zip(MODES, factors, strict=True)
    }
    if weights is not None:
        data["weights"] = (("component",), weights)
    if core is not None:
        data["core"] = (tuple(f"{m}_component" for m in MODES), core)
    return xr.Dataset(data, coords=labels, attrs=attrs)


def _mode_labels(pooled: PooledTensor) -> dict[str, np.ndarray]:
    """Each mode of `MODES`'s labels."""
    return dict(
        zip(
            MODES,
            (pooled.samples, pooled.celltypes, pooled.celltypes, pooled.lr_pairs),
            strict=True,
        )
    )
