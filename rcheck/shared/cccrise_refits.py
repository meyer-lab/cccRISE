"""cccRISE seed refits, keeping the factors (figureR_scriabin_model A:
`cccrise-scriabin-stability --analysis seed_agreement` reads them).

Refits a registry dataset end to end exactly as `run_stability`'s seed axis
does (same fit arguments; compressed RISE; CP on `cp_device`), relabelled into
the core fit's numbering, and saves one .npz per seed: CP weights/factors,
RISE factors, per-cell projections, cell / condition / LR labels, R2X, wall
time and device. Seed 42 is the stability reference; 42001.. are its
seed-axis replicates (`replicate_seed(42, r)`), so their FMS against the
reference can be checked against component_stability_<dataset>.parquet.

Checkpoints: each seed's RISE fit is saved (``seed<k>.rise.npz``: the fields
`fit_rise` writes) before CP, so a CP failure resumes at CP. ``CP_CHUNKED=1``
runs CP through `chunked_cp.chunked_cupy_cp` (cards too small for the full-size
copies). The staged path is
`run_ccc_rise_workflow` step for step.

Usage (ALAD: 42 and 42001-42010; IBD: 42 and 42001-42004, one per process):
  CUDA_VISIBLE_DEVICES=1 uv run --extra gpu python rcheck/shared/cccrise_refits.py \
      alad rcheck/out/cccrise/alad/s1 42 42001 42002 ...
"""

import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import chunked_cp
import numpy as np

from cccrise.ccc_rise import CP_DTYPE, fit_ccc_cp, fit_rise
from cccrise.import_data import add_cond_idxs, import_ligand_receptor_pairs
from cccrise.runners._cli import (
    CP_N_ITER_MAX,
    CP_TOL,
    REPLICATE_COMPRESS,
    RISE_N_ITER_MAX,
    RISE_TOL,
    core_numbering_reference,
)
from cccrise.runners.datasets import DATASETS
from cccrise.validation.stability import relabel_workflow_fit

RANKS = {"alad": (17, 5), "ibd": (60, 6)}


RISE_FIELDS = {
    "uns": ("Pf2_A", "Pf2_B", "Pf2_weights", "r2x_pf2"),
    "varm": ("Pf2_C",),
    "obsm": ("projections", "weighted_projections"),
}


def rise_stage(data, ckpt: Path, rank: int, seed: int, spec):
    """`fit_rise` on `data`, or its saved result from `ckpt` (written
    atomically after a fresh fit)."""
    if ckpt.exists():
        saved = np.load(ckpt)
        for where, keys in RISE_FIELDS.items():
            for k in keys:
                getattr(data, where)[k] = saved[f"{where}.{k}"]
        data.uns["r2x_pf2"] = float(data.uns["r2x_pf2"])
        print(f"seed {seed}: RISE restored from {ckpt.name}", flush=True)
        return data, data.uns["r2x_pf2"]
    data, r2x = fit_rise(
        data,
        rank=rank,
        n_iter_max=RISE_N_ITER_MAX,
        tolerance=RISE_TOL,
        random_state=seed,
        backend=spec.backend,
        compress=REPLICATE_COMPRESS,
    )
    partial = ckpt.with_suffix(".partial.npz")
    np.savez(partial, **{f"{w}.{k}": np.asarray(getattr(data, w)[k])
                         for w, keys in RISE_FIELDS.items() for k in keys})
    partial.replace(ckpt)
    return data, r2x


def refit(data, spec, core, lr_pairs, seed: int, path: Path, rise_ckpt: Path) -> None:
    """One end-to-end refit of `data` (an AnnData with condition indices) at
    `seed`, saved atomically to `path`."""
    rise_rank, cp_rank = RANKS[spec.key]
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    t0 = time.time()
    data, r2x_rise = rise_stage(data, rise_ckpt, rise_rank, seed, spec)
    chunked = os.environ.get("CP_CHUNKED") == "1"
    with chunked_cp.chunked_cupy_cp() if chunked else nullcontext():
        fit = fit_ccc_cp(
            data,
            lr_pairs,
            cp_rank,
            complex_sep="&",
            random_state=seed,
            cp_device=None,
            cp_dtype=CP_DTYPE,
            cp_n_iter_max=CP_N_ITER_MAX,
            cp_tol=CP_TOL,
        )
    fit = fit | {"adata": data, "tensor": None, "r2x_rise": r2x_rise}
    fit = relabel_workflow_fit(fit, core, spec.condition_column)
    a = fit["adata"]
    wall = time.time() - t0
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        cp_weights=fit["cp_weights"],
        **{f"cp_{m}": f for m, f in zip("ABCD", fit["cp_factors"], strict=True)},
        pf2_A=a.uns["Pf2_A"],
        pf2_B=a.uns["Pf2_B"],
        pf2_C=np.asarray(a.varm["Pf2_C"]),
        projections=np.asarray(a.obsm["projections"]),
        cells=np.asarray(a.obs_names, dtype=str),
        conditions=a.obs[spec.condition_column].astype(str).to_numpy(),
        lr_symbols=fit["filtered_lr_pairs"]["interaction_symbol"].to_numpy(str),
        r2x_rise=float(a.uns["r2x_pf2"]),
        r2x_cp=float(fit["r2x_cp"]),
        wall_seconds=wall,
        device=device,
        cp_chunked=chunked,
    )
    partial.replace(path)
    print(f"{spec.key} {path.stem} seed {seed}: {wall:.0f} s on GPU {device}, "
          f"R2X CP {fit['r2x_cp']:.4f}", flush=True)


def setup(dataset: str):
    spec = DATASETS[dataset]
    rise_rank, cp_rank = RANKS[dataset]
    return spec, spec.load(), import_ligand_receptor_pairs(), core_numbering_reference(
        spec, rise_rank, cp_rank
    )


def main(dataset: str, out: Path, seeds: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if not (out / f"seed{s}.npz").exists()]
    if not todo:
        return
    spec, adata, lr_pairs, core = setup(dataset)
    for seed in todo:
        # One seed per process for IBD: a copy of its 57 GB load would OOM
        data = add_cond_idxs(
            adata.copy() if len(todo) > 1 else adata, spec.condition_column
        )
        refit(data, spec, core, lr_pairs, seed, out / f"seed{seed}.npz",
              out / f"seed{seed}.rise.npz")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), [int(s) for s in sys.argv[3:]])
