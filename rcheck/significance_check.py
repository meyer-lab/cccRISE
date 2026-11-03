"""Batched significance (`cccrise.comparators.significance_batched`) on the
GPU against the port and against upstream R.

1. Parity-gate data (`rcheck/out/check_<dataset>`, after `compare.py
   significance`): with the gate's random sets (also fed to R), every
   program's value on the GPU equals `scriabin.module_significance`, and
   matches R's fractions to 1e-12.
2. One real IBD sample of the pilot: GPU equals the CPU batch on the same
   NumPy-drawn sets, for programs spanning sizes; then all of the 40-sample
   merge's programs at 10,000 replicates on the GPU, timed.

Usage: CUDA_VISIBLE_DEVICES=0 uv run --extra gpu python rcheck/significance_check.py
Exits non-zero on any mismatch; results in rcheck/out/significance_check.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cccrise.comparators import scriabin
from cccrise.comparators.significance_batched import sample_values

OUT = Path("rcheck/out")
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def gate(dataset: str) -> None:
    d = OUT / f"check_{dataset}"
    sets = json.loads((d / "random_sets.json").read_text())
    merged = json.loads((d / "r_merged.json").read_text())
    r = pd.read_csv(d / "r_significance.csv")
    for s, per_program in sets.items():
        sim = pd.read_csv(d / f"r_sim_{s}.csv", index_col=0)
        draws = {n: np.array(rs) for n, rs in zip(merged, per_program, strict=True)}
        gpu = sample_values(sim, merged, 0, [0] * len(merged), "gpu", draws)
        port = {
            n: scriabin.module_significance(sim, m, draws[n]) for n, m in merged.items()
        }
        diff = max(abs(gpu[n] - port[n]) for n in merged)
        check(
            f"{dataset} {s}: GPU == port ({len(merged)} programs)",
            diff == 0,
            f"max |diff| {diff:.1e}",
        )
        r_tab = r.drop_duplicates("name").set_index("name")[f"{s}_pval"]
        rd = max(abs(gpu[n] - r_tab[n]) for n in r_tab.index)
        check(
            f"{dataset} {s}: GPU vs R ({len(r_tab)} programs R reports)",
            rd < 1e-12,
            f"max |diff| {rd:.1e}",
        )


def _pilot_sim(sample: str) -> pd.DataFrame:
    d = OUT / "shared/ibd/pilot/programs/sims"
    names = (d / f"{sample}.txt").read_text().splitlines()
    m = np.fromfile(d / f"{sample}.f64").reshape(len(names), len(names), order="F")
    return pd.DataFrame(m, index=names, columns=names)


def ibd_sample() -> None:
    pilot = OUT / "shared/ibd/pilot"
    sample = (pilot / "subset40.txt").read_text().splitlines()[4]
    sim = _pilot_sim(sample)
    merged = {
        k: v if isinstance(v, list) else [v]
        for k, v in json.loads((pilot / "merge40/merged.json").read_text()).items()
    }
    sizes = pd.Series({k: len(v) for k, v in merged.items()}).sort_values()
    testable = sizes[sizes <= sim.shape[0]]
    pick = list(testable.index[np.linspace(0, len(testable) - 1, 10).astype(int)])
    rng = np.random.default_rng(0)
    draws = {
        n: np.stack(
            [
                rng.choice(sim.shape[0], len(merged[n]), replace=False)
                for _ in range(2000)
            ]
        )
        for n in pick
    }
    sub = {n: merged[n] for n in pick}
    gpu = sample_values(sim, sub, 0, [0] * len(sub), "gpu", draws)
    cpu = sample_values(sim, sub, 0, [0] * len(sub), "cpu", draws)
    diff = max(abs(gpu[n] - cpu[n]) for n in sub)
    check(
        f"IBD {sample}: GPU == CPU, 10 programs of size {sizes[pick].min()}-{sizes[pick].max()}, 2,000 sets",
        diff == 0,
        f"max |diff| {diff:.1e}",
    )
    t = time.time()
    seeds = [1111 * 100_000 + j for j in range(len(merged))]
    vals = sample_values(sim, merged, 10_000, seeds, "gpu")
    dt = time.time() - t
    check(
        f"IBD {sample}: all {len(merged)} programs x 10,000 replicates on GPU",
        all(0 <= v <= 1 for v in vals.values()),
        f"{dt:.0f} s; {sum(v <= 0.05 for v in vals.values())} significant",
    )


if __name__ == "__main__":
    for ds in ("alad", "ibd"):
        gate(ds)
    ibd_sample()
    (OUT / "significance_check.json").write_text(json.dumps(RESULTS, indent=1))
    sys.exit(0 if all(r["ok"] for r in RESULTS) else 1)
