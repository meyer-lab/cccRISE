"""Seed-to-seed module stability of Scriabin interaction programs: upstream R
(`seed_stability.R`, its own RNG) vs the Python port on the same samples and
cells, five seeds each. Stability of two runs is
`singlecell_joint.module_stability`. Prints a summary and writes every run
pair to ``data/validation_results/singlecell/scriabin_seed_stability_alad.parquet``
(the RNG-audit panel of `figureS_singlecell`).

    uv run python rcheck/seed_stability.py rcheck/out/seeds
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _common import dataset_inputs
from cccrise.runners.run_singlecell_benchmark import OUT_DIR
from cccrise.validation.singlecell_joint import module_stability

from cccrise.comparators import scriabin

SAMPLES = ("ALAD1-HS1-E1", "ALAD1-HS40-E1")
SEEDS = range(1, 6)
MIN_CELL = 3


def _pairs(sample: str, runs: dict[str, list[dict]]) -> list[dict]:
    """Every pair of runs within a language, and every R-Python pair."""
    rows = [
        {
            "sample": sample,
            "comparison": lang,
            "seed_a": i + 1,
            "seed_b": j + 1,
            "stability": module_stability(a, b),
        }
        for lang, mods in runs.items()
        for (i, a), (j, b) in itertools.combinations(enumerate(mods), 2)
    ]
    rows += [
        {
            "sample": sample,
            "comparison": "R vs Python",
            "seed_a": i + 1,
            "seed_b": j + 1,
            "stability": module_stability(a, b),
        }
        for i, a in enumerate(runs["R"])
        for j, b in enumerate(runs["Python"])
    ]
    return rows


def _r_modules(path: Path) -> dict:
    mods = json.loads(path.read_text())
    return {k: v if isinstance(v, list) else [v] for k, v in mods.items()}


def main() -> None:
    rout = Path(sys.argv[1])
    inp = dataset_inputs("alad")
    names = inp.lr["lr_name"].to_numpy()
    rows = []
    for s in SAMPLES:
        cells = np.flatnonzero(inp.obs["sample"].to_numpy() == s)
        ct = inp.obs["celltype"].to_numpy()[cells]
        counts = pd.Series(ct).value_counts()
        cells = cells[np.isin(ct, counts.index[counts >= MIN_CELL])]
        ct = inp.obs["celltype"].to_numpy()[cells]
        lig, rec = inp.lr_expression(cells)
        py = [
            scriabin.interaction_programs(
                lig, rec, names, ct, np.random.default_rng(seed)
            ).modules
            for seed in SEEDS
        ]
        r = [_r_modules(rout / f"r_{s}_seed{seed}.json") for seed in SEEDS]
        rows += _pairs(s, {"R": r, "Python": py})
        print(
            f"{s}: {len(cells)} cells, {scriabin.n_iterations(len(cells))} subsamples"
        )
    table = pd.DataFrame(rows)
    print(
        table.groupby(["sample", "comparison"])["stability"].agg(["mean", "min", "max"])
    )
    table.to_parquet(OUT_DIR / "scriabin_seed_stability_alad.parquet")


if __name__ == "__main__":
    main()
