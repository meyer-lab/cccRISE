"""S7 parameter sweep in the port (CONCORDANCE_PLAN.md §5.1, S7): Scriabin's
module-discovery settings x seeds on the R-shared inputs, merged programs
compared between seeds (no significance, §8).

Grid: soft-power R^2 cut {0.5, 0.6, 0.7, 0.8} x power cap {3, 6, 12, none}
x ``n.iterate`` {upstream, 32, 128} x ``minClusterSize`` {5, 10, 20} x
``cutHeight`` quantile {0.25, 0.5, 0.75} x merge Jaccard {0.1, 0.15, 0.25,
0.4}. Inputs are the R arm's export (``singlecell_cache/shared/<ds>``:
``lr_expr.mtx``, the LR pairs, cell types), with ip_runs.R's one deviation
(types under 3 cells dropped before subsampling).

Stages (each checkpointed, atomic, resumable):

- ``programs``: one unit per (seed, n.iterate, sample): its averaged
  similarity (`scriabin.ccim_similarity`, the exact factored CCIM
  correlation), the soft power at every (R^2, cap), and the modules at every
  module setting -> ``units/<seed>/<nrep>/<sample>.json``;
- ``merge``: per (seed, module setting, merge Jaccard),
  `scriabin.merge_programs_incremental` over the samples (sorted, as upstream),
  deduplicated by the hash of its input -> ``merged/<hash>.json`` plus
  ``merged/index.csv``;
- ``agree``: per setting, best-match / Hungarian Jaccard of merged programs
  between every seed pair -> ``s7_grid.parquet``.

Usage: uv run python rcheck/shared/s7_grid.py STAGE --dataset alad
         [--seeds 1111 2 3] [--workers N] [--samples K]
(--samples K: the first K samples only, for a pilot; output in s7_pilot/)
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from cccrise.comparators import _wgcna, scriabin

CACHE = Path("singlecell_cache/shared")
OUT = Path("rcheck/out/shared")
R2 = (0.5, 0.6, 0.7, 0.8)
CAPS = (3, 6, 12, None)
NREP = ("upstream", 32, 128)
MIN_SIZE = (5, 10, 20)
CUT_Q = (0.25, 0.5, 0.75)
MERGE_J = (0.1, 0.15, 0.25, 0.4)
BASE = {"r2": 0.6, "cap": 3, "nrep": "upstream", "min_size": 5, "cut_q": 0.5}


def module_settings():
    """Every (r2, cap, min_size, cut_q) combination."""
    return list(itertools.product(R2, CAPS, MIN_SIZE, CUT_Q))


def setting_key(r2, cap, min_size, cut_q) -> str:
    return f"r2={r2}|cap={cap}|min={min_size}|cutq={cut_q}"


# ---- inputs -------------------------------------------------------------


def load_export(dataset: str):
    """(cells x features CSR, feature index, LR table, meta), with the
    MatrixMarket export cached as .npz."""
    d = CACHE / dataset
    npz = d / "lr_expr.npz"
    if not npz.exists():
        m = scipy.io.mmread(d / "lr_expr.mtx").T.tocsr()
        sp.save_npz(d / "lr_expr.partial.npz", m)
        (d / "lr_expr.partial.npz").replace(npz)
    X = sp.load_npz(npz).tocsr()
    features = pd.Index(pd.read_csv(d / "lr_features.csv", header=None)[0])
    lr = pd.read_csv(d / "lr.csv")
    meta = pd.read_csv(d / "meta.csv")
    return X, features, lr, meta


def sample_cells(meta: pd.DataFrame, sample: str) -> np.ndarray:
    """ip_runs.R's cells: types under min.cell dropped when subsampled."""
    rows = np.flatnonzero(meta["sample"].to_numpy() == sample)
    if len(rows) <= scriabin.ITERATE_THRESHOLD:
        return rows
    types = meta["celltype"].to_numpy()[rows]
    counts = pd.Series(types).value_counts()
    return rows[np.isin(types, counts.index[counts >= scriabin.MIN_CELL])]


# ---- programs stage -----------------------------------------------------

_DATA: dict = {}


def _init(dataset: str) -> None:
    X, features, lr, meta = load_export(dataset)
    lig = features.get_indexer(lr["ligand"])
    rec = features.get_indexer(lr["receptor"])
    assert (lig >= 0).all() and (rec >= 0).all()
    _DATA.update(X=X, lig=lig, rec=rec, names=lr["lr_name"].to_numpy(), meta=meta)


def _soft_powers(sim: np.ndarray) -> dict:
    """Soft power at every (R^2, cap), from one fit table."""
    _, table = _wgcna.pick_soft_threshold(sim)
    powers = sorted(_wgcna.DEFAULT_POWERS)
    out = {}
    for r2, cap in itertools.product(R2, CAPS):
        passed = np.flatnonzero(table[:, 0] > r2)
        p = powers[passed[0]] if len(passed) else None
        out[(r2, cap)] = 1 if p is None or (cap is not None and p > cap) else p
    return out


def _modules(sim: pd.DataFrame, power: int, min_size: int, cut_q: float, cache: dict):
    if power not in cache:
        adj = sim.to_numpy() ** power
        cache[power] = _wgcna.hclust_complete(1.0 - _wgcna.signed_tom(adj))
    tree = cache[power]
    labels = _wgcna.cutree_dynamic_tree(
        tree, float(np.quantile(tree.height, cut_q)), min_size
    )
    names = sim.index.to_numpy()
    order = [lab for lab in pd.unique(labels) if lab != 0]
    return {f"IP-{i + 1}": list(names[labels == lab]) for i, lab in enumerate(order)}


def run_unit(args: tuple) -> str:
    seed, nrep, sample, idx, path = args
    path = Path(path)
    if path.exists():
        return ""
    t0 = time.time()
    meta = _DATA["meta"]
    rows = sample_cells(meta, sample)
    Xs = _DATA["X"][rows]
    lig = Xs[:, _DATA["lig"]].toarray()
    rec = Xs[:, _DATA["rec"]].toarray()
    types = meta["celltype"].to_numpy()[rows]
    n_code = 0 if nrep == "upstream" else int(nrep)
    rng = np.random.default_rng(np.random.SeedSequence([seed, idx, n_code]))
    settings = scriabin.ProgramSettings(
        n_iterate=None if nrep == "upstream" else int(nrep)
    )
    subs = scriabin.draw_subsamples(types, rng, settings)
    sim = scriabin.averaged_similarity(lig, rec, _DATA["names"], subs, factored=True)
    powers = _soft_powers(sim.to_numpy())
    cache: dict = {}
    mods = {
        setting_key(r2, cap, m, q): _modules(sim, powers[(r2, cap)], m, q, cache)
        for r2, cap, m, q in module_settings()
    }
    rec_out = {
        "n_cells": len(rows),
        "n_lr": sim.shape[0],
        "n_subsamples": len(subs),
        "soft_power": {f"r2={r}|cap={c}": int(p) for (r, c), p in powers.items()},
        "modules": mods,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(rec_out))
    partial.replace(path)
    return (
        f"{sample} seed {seed} nrep {nrep}: {len(rows)} cells, {time.time() - t0:.1f} s"
    )


def stage_programs(dataset, seeds, workers, out, samples):
    units = [
        (seed, nrep, s, i, str(out / "units" / str(seed) / str(nrep) / f"{s}.json"))
        for seed in seeds
        for nrep in NREP
        for i, s in enumerate(samples)
    ]
    todo = [u for u in units if not Path(u[4]).exists()]
    print(f"{len(todo)} of {len(units)} units to run, {workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dataset,)) as ex:
        for k, msg in enumerate(ex.map(run_unit, todo, chunksize=1)):
            if k % 25 == 0 or k == len(todo) - 1:
                print(
                    f"{time.strftime('%H:%M:%S')} [{k + 1}/{len(todo)}] {msg} "
                    f"({time.time() - t0:.0f} s)",
                    flush=True,
                )


# ---- merge stage --------------------------------------------------------


def _per_sample(out: Path, seed, nrep, samples, key: str) -> dict:
    per = {}
    for s in samples:
        rec = json.loads(
            (out / "units" / str(seed) / str(nrep) / f"{s}.json").read_text()
        )
        per[s] = rec["modules"][key]
    return per


def _merge_one(args: tuple) -> str:
    per_sample, thr, path = args
    path = Path(path)
    if path.exists():
        return ""
    merged = scriabin.merge_programs_incremental(per_sample, thr)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(merged))
    partial.replace(path)
    return path.stem


def _unit_cache(out: Path, seed, nrep, samples) -> dict:
    return {
        s: json.loads((out / "units" / str(seed) / str(nrep) / f"{s}.json").read_text())
        for s in samples
    }


def stage_merge(seeds, workers, out, samples):
    mdir = out / "merged"
    mdir.mkdir(parents=True, exist_ok=True)
    index, jobs, queued = [], [], set()
    for seed, nrep in itertools.product(seeds, NREP):
        units = _unit_cache(out, seed, nrep, samples)
        for setting in module_settings():
            key = setting_key(*setting)
            per = {s: units[s]["modules"][key] for s in samples}
            digest = hashlib.sha1(json.dumps(per, sort_keys=True).encode()).hexdigest()
            for thr in MERGE_J:
                h = f"{digest[:20]}_{thr}"
                index.append(
                    {
                        "seed": seed,
                        "nrep": str(nrep),
                        "setting": key,
                        "merge_j": thr,
                        "hash": h,
                    }
                )
                if h not in queued and not (mdir / f"{h}.json").exists():
                    queued.add(h)
                    jobs.append((per, thr, str(mdir / f"{h}.json")))
    print(
        f"{len(index)} (seed, setting, merge) cells, {len(jobs)} distinct merges "
        f"to run",
        flush=True,
    )
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(_merge_one, jobs, chunksize=1)):
            if k % 100 == 0 or k == len(jobs) - 1:
                print(
                    f"{time.strftime('%H:%M:%S')} [{k + 1}/{len(jobs)}] "
                    f"({time.time() - t0:.0f} s)",
                    flush=True,
                )
    pd.DataFrame(index).to_csv(mdir / "index.partial.csv", index=False)
    (mdir / "index.partial.csv").replace(mdir / "index.csv")


# ---- agreement stage ----------------------------------------------------


def membership(a: list[set], b: list[set]) -> tuple[float, float]:
    """Best-match Jaccard (mean of both directions) and Hungarian-matched
    Jaccard, as S1 (`scratchpad/s1_analysis.py`)."""
    if not a or not b:
        return float(a == b), float(a == b)
    universe = sorted(set().union(*a, *b))
    pos = {x: i for i, x in enumerate(universe)}
    ma = np.zeros((len(a), len(universe)))
    mb = np.zeros((len(b), len(universe)))
    for i, s in enumerate(a):
        ma[i, [pos[x] for x in s]] = 1
    for i, s in enumerate(b):
        mb[i, [pos[x] for x in s]] = 1
    inter = ma @ mb.T
    j = inter / (ma.sum(1)[:, None] + mb.sum(1)[None, :] - inter)
    best = (j.max(axis=1).mean() + j.max(axis=0).mean()) / 2
    r, c = linear_sum_assignment(-j)
    return float(best), float(j[r, c].sum() / max(j.shape))


def stage_agree(seeds, out):
    index = pd.read_csv(out / "merged" / "index.csv")
    sets = {}

    def programs(h):
        if h not in sets:
            m = json.loads((out / "merged" / f"{h}.json").read_text())
            sets[h] = [set(v if isinstance(v, list) else [v]) for v in m.values()]
        return sets[h]

    rows = []
    for (nrep, setting, thr), g in index.groupby(["nrep", "setting", "merge_j"]):
        by_seed = dict(zip(g["seed"], g["hash"], strict=True))
        bm, hu = [], []
        for sa, sb in itertools.combinations(sorted(by_seed), 2):
            b, h = membership(programs(by_seed[sa]), programs(by_seed[sb]))
            bm.append(b)
            hu.append(h)
        n_prog = [len(programs(by_seed[s])) for s in by_seed]
        share = [
            max(len(p) for p in programs(by_seed[s]))
            / len(set().union(*programs(by_seed[s])))
            if programs(by_seed[s])
            else np.nan
            for s in by_seed
        ]
        parts = dict(kv.split("=") for kv in setting.split("|"))
        rows.append(
            {
                "nrep": nrep,
                "r2": float(parts["r2"]),
                "cap": None if parts["cap"] == "None" else int(parts["cap"]),
                "min_size": int(parts["min"]),
                "cut_q": float(parts["cutq"]),
                "merge_j": thr,
                "best_match": np.mean(bm),
                "best_match_min": np.min(bm),
                "best_match_max": np.max(bm),
                "hungarian": np.mean(hu),
                "n_programs": np.mean(n_prog),
                "largest_share": np.mean(share),
            }
        )
    table = pd.DataFrame(rows).sort_values("best_match", ascending=False)
    table.to_parquet(out / "s7_grid.partial.parquet")
    (out / "s7_grid.partial.parquet").replace(out / "s7_grid.parquet")
    is_base = (
        (table.nrep == "upstream")
        & (table.r2 == BASE["r2"])
        & (table.cap == BASE["cap"])
        & (table.min_size == BASE["min_size"])
        & (table.cut_q == BASE["cut_q"])
        & (table.merge_j == 0.15)
    )
    print(table.head(10).to_string(index=False))
    print("base:", table[is_base].to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=["programs", "merge", "agree"])
    p.add_argument("--dataset", required=True, choices=["alad", "ibd"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1111, 2, 3])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--samples", type=int)
    a = p.parse_args()
    meta = pd.read_csv(CACHE / a.dataset / "meta.csv", usecols=["sample"])
    samples = sorted(meta["sample"].unique())
    out = OUT / a.dataset / "s7"
    if a.samples:
        samples, out = samples[: a.samples], OUT / a.dataset / "s7_pilot"
    if a.stage == "programs":
        stage_programs(a.dataset, a.seeds, a.workers, out, samples)
    elif a.stage == "merge":
        stage_merge(a.seeds, a.workers, out, samples)
    else:
        stage_agree(a.seeds, out)


if __name__ == "__main__":
    sys.exit(main())
