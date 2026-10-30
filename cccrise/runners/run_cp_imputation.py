#!/usr/bin/env python
"""
Runner: imputation R2X of CP models across ranks, the held-out companion to
fit R2X in the CP rank panel.

The communication tensor's entries (of the dataset's canonical RISE fit) are
split into `CP_IMPUTATION_FOLDS` random folds (`entry_folds`), shared by every
rank. For each fold and rank, the fold's entries are zeroed and treated as
missing, CP is fit from the production start (the seeded SVD initialisation
of `run_cpd_rank_rule`, start 0) by masked ALS (`masked_parafac`), and the
fold is scored by the R2X of the model's values at its entries:
``1 - sum((x - x_hat)^2) / sum(x^2)`` over the held-out entries alone. Fits
are uncompressed (masks hold out entries of the original tensor, which a
CANDELINC core mixes) and run in float64, on the GPU with ``--backend cupy``.
One checkpoint per (rank, fold), so a run resumes.

Writes ``data/rank_selection_results/cp_imputation_<dataset>.parquet``, one
row per (rank, fold).

    uv run python -m cccrise.runners.run_cp_imputation --dataset alad --backend cupy
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors import clean_memory
from ..rank_selection import CP_IMPUTATION_FOLDS, entry_folds, masked_parafac
from ._cli import (
    CP_N_ITER_MAX,
    CP_TOL,
    add_dataset_argument,
    dataset_metadata,
    parse_ranks,
    resolve_dataset,
)
from .datasets import RANK_SELECTION_DIR, DatasetSpec
from .provenance import REPO_ROOT, build_metadata, sha256_file, write_metadata
from .run_cpd_rank_rule import interaction_tensor, rise_digest, start_factors

CHECKPOINT_DIR = REPO_ROOT / "output" / "cache" / "cp_imputation"
HEADER = """\
# CP imputation R2X, one row per (rank, fold): the R2X of a masked-ALS fit's
# values at the fold's held-out entries; TopShare is the share of its squared
# error in its 10 largest entries.
"""


#: Held-out entries whose share of the squared error is reported (`TopShare`).
TOP_ENTRIES = 10


def heldout_r2x(actual: np.ndarray, imputed: np.ndarray) -> float:
    """R2X of `imputed` against `actual` over the held-out entries."""
    return float(1.0 - np.sum((actual - imputed) ** 2) / np.sum(actual**2))


def top_share(actual: np.ndarray, imputed: np.ndarray) -> float:
    """Share of the held-out squared error in its `TOP_ENTRIES` largest
    entries: how far a handful of entries decide the fold's R2X."""
    errors = (actual - imputed) ** 2
    return float(np.sort(errors)[-TOP_ENTRIES:].sum() / errors.sum())


def fit_fold(
    observed: np.ndarray, held: np.ndarray, rank: int, seed: int, backend: str
) -> tuple[np.ndarray, int]:
    """Masked ALS on `observed` (zeros at `held`) from the production start;
    the model's values at `held` (NumPy) and the sweeps run."""
    init = start_factors(observed, rank, "svd", seed)
    if backend == "cupy":
        import cupy

        device = cupy.asarray(observed)
    else:
        device = observed.copy()
    try:
        _weights, _factors, imputed, sweeps = masked_parafac(
            device, held, (init.weights, init.factors), CP_N_ITER_MAX, CP_TOL
        )
        imputed = imputed.get() if backend == "cupy" else imputed
    finally:
        del device
        if backend == "cupy":
            clean_memory()
    return np.asarray(imputed, dtype=float), sweeps


def run(
    spec: DatasetSpec,
    ranks: list[int],
    n_folds: int,
    backend: str,
    random_state: int,
    tensor_cache: Path | None,
) -> pd.DataFrame:
    print(
        f"=== CP imputation: {spec.label}  ranks={ranks}  folds={n_folds}  backend={backend}"
    )
    tensor = interaction_tensor(spec, tensor_cache)
    folds = entry_folds(tensor.shape, n_folds, random_state)
    checkpoints = (
        CHECKPOINT_DIR / f"{spec.output_key}_{rise_digest(spec.factors_path())}"
    )
    checkpoints.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(n_folds):
        held_mask = folds == fold
        held = np.flatnonzero(held_mask)
        actual = tensor.reshape(-1)[held]
        observed = np.where(held_mask, 0.0, tensor)
        del held_mask
        for rank in ranks:
            path = checkpoints / f"r{rank}_f{fold}.json"
            if not path.exists():
                start = time.time()
                imputed, sweeps = fit_fold(observed, held, rank, random_state, backend)
                row = {"Rank": rank, "Fold": fold, "ImputationR2X": heldout_r2x(actual, imputed),
                       "TopShare": top_share(actual, imputed), "Sweeps": sweeps,
                       "HeldOut": int(held.size)}  # fmt: skip
                path.write_text(json.dumps(row))
                print(f"   fold {fold} rank {rank}: imputation R2X {row['ImputationR2X']:.4f}, "
                      f"{sweeps} sweeps, {time.time() - start:.0f} s", flush=True)  # fmt: skip
            rows.append(json.loads(path.read_text()))
        del observed
    table = pd.DataFrame(rows)
    out = RANK_SELECTION_DIR / f"cp_imputation_{spec.output_key}.parquet"
    table.to_parquet(out, index=False)
    metadata = build_metadata(dataset_metadata(spec) | {
        "factors_sha256": sha256_file(spec.factors_path()),
        "tensor_source": str(tensor_cache) if tensor_cache else "rebuilt from the factors",
        "ranks": ranks, "folds": n_folds, "random_state": random_state,
        "n_iter_max": CP_N_ITER_MAX, "tol": CP_TOL, "backend": backend,
        "start": "production SVD start 0 on the zero-filled tensor",
    })  # fmt: skip
    write_metadata(out, metadata, header=HEADER)
    print(f"wrote {out}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_dataset_argument(parser)
    parser.add_argument("--ranks", type=str, default="1:18")
    parser.add_argument("--folds", type=int, default=CP_IMPUTATION_FOLDS)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tensor-cache", type=Path, default=None,
                        help="A saved .npy of the communication tensor, used instead of rebuilding it.")  # fmt: skip
    args = parser.parse_args()
    run(resolve_dataset(args), parse_ranks(args.ranks), args.folds, args.backend,
        args.random_state, args.tensor_cache)  # fmt: skip


if __name__ == "__main__":
    main()
