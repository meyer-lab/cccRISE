"""Cell subsets shared by Scriabin (R) and cccRISE for S2 and S3
(CONCORDANCE_PLAN.md §5.1), so both methods see the same cells.

- ``s2``: per (fraction, draw), each sample keeps ``round(f * n)`` of its
  cells, drawn uniformly without replacement (NumPy, seeded per
  (fraction, draw, sample)). ALAD: 0.9 / 0.75 / 0.5 x 3 draws; IBD:
  0.75 / 0.5 x 2 draws. Writes ``s2/subsets/f<f>_d<d>.txt`` (kept cells).
- ``s3``: for the chosen samples, 10 evenly spaced 0-based cell positions
  (the export's cell order, which `ip_runs.R drop` indexes) and their cell
  names -> ``s3/drops.csv``.

Usage: uv run python rcheck/shared/cell_subsets.py {s2,s3} --dataset alad
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("singlecell_cache/shared")
OUT = Path("rcheck/out/shared")
S2 = {"alad": ((0.9, 0.75, 0.5), 3), "ibd": ((0.75, 0.5), 2)}
#: ALAD: every sample <= 300 cells (no subsampling upstream) plus three larger,
#: spanning size; IBD: six spanning size (most are at the 2,000-cell cap).
S3 = {
    "alad": [
        "ALAD1-HS58-E1",
        "ALAD1-HS16-E1",
        "ALAD1-HS62-E1",
        "ALAD1-HS26-E1",
        "ALAD1-HS27-E1",
        "ALAD1-HS19-E1",
        "ALAD1-HS3-E1",
        "ALAD1-HS40-E1",
        "ALAD1-HS72-E1",
    ],
}
N_DROPS = 10


def s2(dataset: str, meta: pd.DataFrame) -> None:
    fracs, draws = S2[dataset]
    out = OUT / dataset / "s2" / "subsets"
    out.mkdir(parents=True, exist_ok=True)
    samples = sorted(meta["sample"].unique())
    for f in fracs:
        for d in range(1, draws + 1):
            keep = []
            for i, s in enumerate(samples):
                cells = meta.loc[meta["sample"] == s, "cell"].to_numpy()
                rng = np.random.default_rng([round(f * 100), d, i])
                pick = rng.choice(len(cells), round(f * len(cells)), replace=False)
                keep.extend(cells[np.sort(pick)])
            (out / f"f{f}_d{d}.txt").write_text("\n".join(keep) + "\n")
            print(f"f{f}_d{d}: {len(keep)} of {len(meta)} cells")


def s3(dataset: str, meta: pd.DataFrame, samples: list[str]) -> None:
    rows = []
    for s in samples:
        cells = meta.loc[meta["sample"] == s, "cell"].to_numpy()
        for idx in np.unique(
            np.round(np.linspace(0, len(cells) - 1, N_DROPS)).astype(int)
        ):
            rows.append(
                {
                    "sample": s,
                    "n_cells": len(cells),
                    "idx": int(idx),
                    "cell": cells[idx],
                }
            )
    out = OUT / dataset / "s3"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "drops.csv", index=False)
    print(pd.DataFrame(rows).groupby("sample").n_cells.first().to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("test", choices=["s2", "s3"])
    p.add_argument("--dataset", required=True, choices=["alad", "ibd"])
    p.add_argument("--samples", nargs="+", help="S3 samples (default: the list above)")
    a = p.parse_args()
    meta = pd.read_csv(CACHE / a.dataset / "meta.csv")
    if a.test == "s2":
        s2(a.dataset, meta)
    else:
        s3(a.dataset, meta, a.samples or S3[a.dataset])


if __name__ == "__main__":
    main()
