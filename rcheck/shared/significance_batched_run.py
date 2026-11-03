"""``InteractionProgramSignificance`` for a whole S1 run, batched on the GPU
(`cccrise.comparators.significance_batched`), where upstream R is
impractical (IBD: ~1,400 CPU-h per seed; CONCORDANCE_PLAN.md §10).

Reads RUN_DIR's ``merged.json`` and each sample's averaged similarity
(``sims/``, from `export_sims.R`); writes, as `merge_sig.R significance`
does, one checkpoint per sample (``significance_batched/<sample>.json``:
program -> value) and then ``kept.json`` (significant, value <= 0.05, in at
least one sample, with more than one ligand and receptor:
`scriabin.kept_programs`), ``significance_values.csv`` and
``significant_in.csv``. Random sets are seeded per (run seed, sample, program).
GPU = port = R on identical sets (`rcheck/significance_check.py`).

Usage: CUDA_VISIBLE_DEVICES=0 uv run --extra gpu python \
    rcheck/shared/significance_batched_run.py RUN_DIR SEED [--replicates 10000]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cccrise.comparators import scriabin
from cccrise.comparators.significance_batched import sample_values


def _sim(run: Path, sample: str) -> pd.DataFrame:
    names = (run / "sims" / f"{sample}.txt").read_text().splitlines()
    m = np.fromfile(run / "sims" / f"{sample}.f64").reshape(
        len(names), len(names), order="F"
    )
    return pd.DataFrame(m, index=names, columns=names)


def main(run: Path, seed: int, replicates: int, device: str) -> None:
    merged = {
        k: v if isinstance(v, list) else [v]
        for k, v in json.loads((run / "merged.json").read_text()).items()
    }
    samples = sorted(f.stem for f in (run / "sims").glob("*.txt"))
    out = run / "significance_batched"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    for i, s in enumerate(samples):
        if (out / f"{s}.json").exists():
            continue
        seeds = [
            int(np.random.SeedSequence([seed, i, j]).generate_state(1)[0])
            for j in range(len(merged))
        ]
        values = sample_values(_sim(run, s), merged, replicates, seeds, device)
        partial = out / f"{s}.json.partial"
        partial.write_text(json.dumps(values))
        partial.replace(out / f"{s}.json")
        print(
            f"{time.strftime('%H:%M:%S')} {s} ({i + 1}/{len(samples)}): "
            f"{sum(v <= 0.05 for v in values.values())} significant, "
            f"{time.time() - t0:.0f} s elapsed",
            flush=True,
        )
    table = pd.DataFrame(
        {s: json.loads((out / f"{s}.json").read_text()) for s in samples}
    )
    table = table.loc[list(merged)]
    table.to_csv(run / "significance_values.csv")
    (table <= 0.05).to_csv(run / "significant_in.csv")
    kept = scriabin.kept_programs(merged, table)
    (run / "kept.json").write_text(json.dumps(kept))
    print(
        f"{len(kept)} of {len(merged)} programs kept ({replicates} replicates, {device})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("seed", type=int)
    p.add_argument("--replicates", type=int, default=10_000)
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    a = p.parse_args()
    main(a.run, a.seed, a.replicates, a.device)
