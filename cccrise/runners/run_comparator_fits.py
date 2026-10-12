#!/usr/bin/env python
"""
Runner: own-rank comparator fits on labellings built from the pseudobulk cache.

`run_pseudobulk_cache` stores per-(sample, fine type) sums; this builds the
pseudobulk tensor of each requested labelling from them
(`comparators.pseudobulk_cache.labelled_matrices`, then the same LR scoring as
`benchmark.pseudobulk_tensor`) and fits every comparator with its own rank
selection (`benchmark.fit_comparators`). Labellings:

- ``coarse``: the coarse annotation (the dataset's `celltype_column`);
- ``fine``: every fine type;
- ``s<n>r<k>``: `n` fine types drawn uniformly at random (seed
  ``1000 * n + k``) broken out of their coarse type, every other fine type
  staying in its coarse type -- the granularity steps between the two.

Each labelling's fits, with the published cccRISE fit first, are written as
``benchmark_fits_<dataset><suffix>.nc`` (suffix ``""`` for coarse, ``_fine``,
``_s<n>r<k>``), the file `run_comparator_agreement` and `run_lr_concordance`
read; fits are checkpointed under ``output/cache/benchmark/`` and a labelling
whose file already holds fits with the same settings is skipped.

    uv run python -m cccrise.runners.run_comparator_fits --dataset ibd --levels coarse
    uv run python -m cccrise.runners.run_comparator_fits --dataset ibd \\
        --levels s12 s24 s36 --repeats 5 --tc2c-backend cupy
"""

import argparse
import hashlib
import json
import re
import time
from typing import Any

import numpy as np

from ..benchmark import (
    STACCATO_VAREXP,
    TC2C_ELBOW_RUNS,
    TC2C_UPPER_RANK,
    fit_comparators,
    fit_settings,
    load_fits,
    save_fits,
)
from ..ccc_rise import calc_communication_score_pseudobulk
from ..comparators.pseudobulk_cache import labelled_matrices
from ..factors import ensure_decompressed
from ..import_data import import_ligand_receptor_pairs
from ._cli import add_dataset_argument, resolve_dataset
from .datasets import VALIDATION_DIR, DatasetSpec
from .provenance import sha256_file
from .run_benchmark_anova import CHECKPOINT_DIR, published_cccrise
from .run_pseudobulk_cache import cache_path

SAMPLED = re.compile(r"^s(\d+)$")
#: Settings a labelling's fits depend on (the backend does not count).
KEY_SETTINGS = (
    "factors_sha256",
    "tc2c_upper_rank",
    "tc2c_elbow_runs",
    "staccato_varexp",
    "pb_max_rank",
    "random_state",
)


def labelling(cache: dict, level: str, repeat: int = 0) -> tuple[str, dict, list]:
    """The suffix, fine type -> group mapping and broken-out fine types of
    one labelling."""
    fine = [str(t) for t in cache["fine_types"]]
    coarse = dict(zip(fine, map(str, cache["fine_to_coarse"]), strict=True))
    if level == "coarse":
        return "", coarse, []
    if level == "fine":
        return "_fine", {t: t for t in fine}, fine
    n = int(SAMPLED.match(level).group(1))
    rng = np.random.default_rng(1000 * n + repeat)
    chosen = sorted(rng.choice(fine, size=n, replace=False).tolist())
    return f"_s{n}r{repeat}", coarse | {t: t for t in chosen}, chosen


def labelled_tensor(cache: dict, mapping: dict, lr_pairs, complex_sep="&"):
    """A labelling's pseudobulk tensor and its description."""
    matrices, groups = labelled_matrices(
        cache["counts"], cache["sums"], cache["fine_types"], cache["genes"], mapping
    )
    tensor, filtered = calc_communication_score_pseudobulk(
        matrices, lr_pairs=lr_pairs, complex_sep=complex_sep
    )
    return tensor, {
        "tensor_shape": list(tensor.shape),
        "celltypes": [str(g) for g in groups],
        "n_lr_pairs": len(filtered),
        "lr_pairs": filtered["interaction_symbol"].astype(str).tolist(),
    }


def _fit_one(spec, cache, cccrise_fit, suffix, mapping, chosen, settings) -> None:
    path = VALIDATION_DIR / spec.benchmark_fits_filename(suffix)
    key = {
        "dataset": spec.output_key,
        "labelling": suffix or "_coarse",
        "broken_out": json.dumps(chosen),
        **{k: settings[k] for k in KEY_SETTINGS},
        **fit_settings(),
    }
    conditions = np.asarray(cache["samples"])
    if load_fits(path, conditions, key) is not None:
        print(f">> {suffix or 'coarse'}: fits already cached, skipped")
        return
    start = time.time()
    tensor, info = labelled_tensor(cache, mapping, settings["lr_pairs"])
    print(f">> {suffix or 'coarse'}: tensor {tensor.shape}")
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    fits = fit_comparators(
        tensor,
        rank_mode="own",
        shared_rank=int(cccrise_fit.details["cp_rank"]),
        tc2c_upper_rank=settings["tc2c_upper_rank"],
        tc2c_elbow_runs=settings["tc2c_elbow_runs"],
        staccato_varexp=settings["staccato_varexp"],
        random_state=settings["random_state"],
        tc2c_backend=settings["tc2c_backend"],
        pb_max_rank=settings["pb_max_rank"],
        checkpoint_dir=CHECKPOINT_DIR / digest / "own",
    )
    info["broken_out"] = chosen
    save_fits(path, [cccrise_fit, *fits], conditions, key, info)
    ranks = ", ".join(f"{f.method} {f.rank}" for f in fits)
    print(f"   {ranks}; {time.time() - start:.0f} s; wrote {path.name}")


def run(spec: DatasetSpec, levels: list[str], repeats: int, **settings: Any) -> None:
    cache = dict(np.load(cache_path(spec.output_key), allow_pickle=False))
    factors = ensure_decompressed(spec.factors_path())
    cccrise_fit, _ = published_cccrise(factors, np.asarray(cache["samples"]))
    settings |= {
        "factors_sha256": sha256_file(factors),
        "lr_pairs": import_ligand_receptor_pairs(),
    }
    for level in levels:
        for repeat in range(repeats if SAMPLED.match(level) else 1):
            suffix, mapping, chosen = labelling(cache, level, repeat)
            _fit_one(spec, cache, cccrise_fit, suffix, mapping, chosen, settings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_dataset_argument(parser)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tc2c-backend", default="numpy", choices=["numpy", "cupy"])
    parser.add_argument("--tc2c-upper-rank", type=int, default=TC2C_UPPER_RANK)
    parser.add_argument("--tc2c-elbow-runs", type=int, default=TC2C_ELBOW_RUNS)
    parser.add_argument("--staccato-varexp", type=float, default=STACCATO_VAREXP)
    parser.add_argument("--pb-max-rank", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    run(
        resolve_dataset(args),
        args.levels,
        args.repeats,
        tc2c_backend=args.tc2c_backend,
        tc2c_upper_rank=args.tc2c_upper_rank,
        tc2c_elbow_runs=args.tc2c_elbow_runs,
        staccato_varexp=args.staccato_varexp,
        pb_max_rank=args.pb_max_rank,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
