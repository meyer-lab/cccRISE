"""S3 (one cell, no randomness) in the port: `InteractionPrograms` on each
sample's full CCIM (`scriabin.ccim_similarity`, exact) with all cells and with
each listed cell left out, as `ip_runs.R drop` does in R.

Two uses:
- the R samples of ``s3/drops.csv``: port modules against R's for every run
  (equality check, so the port's S3 speaks for upstream);
- IBD's 2,000-cell samples, whose full CCIM R cannot hold under 48 GB: six
  spanning the LR-pair count, 10 evenly spaced cells each
  (``s3/drops_port.csv``).

Writes ``s3/port/<sample>.json`` ({"-1": modules, "<idx>": modules, ...,
"shift_<idx>": max |similarity change|}), one checkpoint per sample.

Usage: uv run python rcheck/shared/s3_port.py --dataset ibd [--big 6] [--setting LABEL]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7_grid

from cccrise.comparators import scriabin

OUT = Path("rcheck/out/shared")
N_DROPS = 10


def _similarity(lig, rec, names, rows) -> pd.DataFrame:
    keep, sim = scriabin.ccim_similarity(lig[rows], rec[rows])
    sim = np.clip(sim, 0.0, 1.0)
    return pd.DataFrame(sim, index=names[keep], columns=names[keep])


def run_sample(
    sample: str,
    idxs: list[int],
    path: Path,
    settings: scriabin.ProgramSettings = scriabin.UPSTREAM,
) -> None:
    if path.exists():
        return
    d = s7_grid._DATA
    rows = np.flatnonzero(d["meta"]["sample"].to_numpy() == sample)
    Xs = d["X"][rows]
    lig, rec = Xs[:, d["lig"]].toarray(), Xs[:, d["rec"]].toarray()
    base = _similarity(lig, rec, d["names"], np.arange(len(rows)))
    out = {"-1": scriabin.modules_from_similarity(base, settings).modules}
    for idx in idxs:
        sim = _similarity(lig, rec, d["names"], np.delete(np.arange(len(rows)), idx))
        out[str(idx)] = scriabin.modules_from_similarity(sim, settings).modules
        shared = sim.index.intersection(base.index)
        delta = sim.loc[shared, shared].to_numpy() - base.loc[shared, shared].to_numpy()
        out[f"shift_{idx}"] = float(np.nanmax(np.abs(delta)))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(out))
    partial.replace(path)
    print(f"{sample}: {len(rows)} cells, {len(idxs)} drops", flush=True)


def big_samples(dataset: str, k: int) -> pd.DataFrame:
    """k samples at the 2,000-cell cap spanning the LR pairs measured (S1
    seed 1111), with N_DROPS evenly spaced cells each."""
    meta = s7_grid._DATA["meta"]
    s1 = OUT / dataset / "s1" / "seed1111"
    sizes = meta["sample"].value_counts()
    cap = sizes[sizes == sizes.max()].index
    n_lr = pd.Series(
        {s: json.loads((s1 / f"{s}.json").read_text())["n_lr"] for s in cap}
    ).sort_values()
    pick = n_lr.index[np.linspace(0, len(n_lr) - 1, k).astype(int)]
    rows = []
    for s in pick:
        n = int(sizes[s])
        for idx in np.unique(np.round(np.linspace(0, n - 1, N_DROPS)).astype(int)):
            rows.append({"sample": s, "n_cells": n, "idx": int(idx)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, choices=["alad", "ibd"])
    p.add_argument("--big", type=int, default=0)
    p.add_argument(
        "--setting",
        help="label in s7_r/settings.json: module settings (S3 at an S7 "
        "setting; output in s3/port_<label>/)",
    )
    a = p.parse_args()
    s7_grid._init(a.dataset)
    root = OUT / a.dataset / "s3"
    settings, port = scriabin.UPSTREAM, root / "port"
    if a.setting:
        cfg = json.loads((OUT / a.dataset / "s7_r" / "settings.json").read_text())
        c = cfg[a.setting]
        settings = scriabin.ProgramSettings(
            r2_cutoff=c["r2_cutoff"],
            max_soft_power=c["max_soft_power"],
            min_module_size=c["min_size"],
            cut_height_quantile=c["tree_cut_quantile"],
        )
        port = root / f"port_{a.setting}"
    tables = [pd.read_csv(root / "drops.csv")]
    if a.big:
        big = big_samples(a.dataset, a.big)
        big.to_csv(root / "drops_port.csv", index=False)
        tables.append(big)
    for s, g in pd.concat(tables).groupby("sample", sort=False):
        run_sample(s, g["idx"].tolist(), port / f"{s}.json", settings)


if __name__ == "__main__":
    main()
