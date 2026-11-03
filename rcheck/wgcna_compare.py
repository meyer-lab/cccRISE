"""WGCNA / dynamicTreeCut / hclust ports against R, at upstream's settings
and at non-default module-discovery settings (the S7 knobs that act on a
given similarity).

Usage:
  uv run python rcheck/wgcna_compare.py <dir> write [<check_dir> ...]
  Rscript rcheck/wgcna_check.R <dir>
  uv run python rcheck/wgcna_compare.py <dir> compare

`write` saves similarity matrices: a flat one (LR correlations near 0, as
in the ALAD networks), a block-structured one, and, for each `check_dir`
written by `export_inputs.py`, the port's averaged similarity of its first
sample (R's own similarity is checked by `compare.py`); plus `settings.csv`.
`compare` checks the soft-power table, TOM, tree, modules and connectivity,
and exits non-zero on any failure.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from cccrise.comparators import _wgcna, scriabin
from cccrise.comparators.inputs import entity_expression

#: Upstream first, then one change per knob, then combinations the S7 grid
#: visits; NaN = R's NA (no cap / pick the power).
SETTINGS = pd.DataFrame(
    [
        (0.6, 3, np.nan, 5, 0.5),
        (0.5, 3, np.nan, 5, 0.5),
        (0.8, np.nan, np.nan, 5, 0.5),
        (0.6, 12, np.nan, 5, 0.5),
        (0.6, 3, 6, 5, 0.5),
        (0.6, 3, np.nan, 10, 0.5),
        (0.6, 3, np.nan, 20, 0.25),
        (0.6, 3, np.nan, 5, 0.75),
        (0.7, 6, np.nan, 10, 0.25),
    ],
    columns=[
        "r2_cutoff",
        "max_soft_power",
        "soft_power",
        "min_module_size",
        "cut_height_quantile",
    ],
)
TOL = 1e-8
RESULTS: list[dict] = []


def settings(row: pd.Series) -> scriabin.ProgramSettings:
    def opt(v: float) -> int | None:
        return None if np.isnan(v) else int(v)

    return scriabin.ProgramSettings(
        r2_cutoff=float(row.r2_cutoff),
        max_soft_power=opt(row.max_soft_power),
        soft_power=opt(row.soft_power),
        min_module_size=int(row.min_module_size),
        cut_height_quantile=float(row.cut_height_quantile),
    )


def _block_similarity(n_lr: int = 60, n_blocks: int = 3, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(500, n_blocks))
    member = np.arange(n_lr) % n_blocks
    x = factors[:, member] + rng.normal(scale=1.5, size=(500, n_lr))
    return 0.5 + 0.5 * np.corrcoef(x, rowvar=False)


def _sample_similarity(check_dir: Path) -> np.ndarray:
    """The port's averaged similarity of the first sample in `check_dir`."""
    X = scipy.io.mmread(check_dir / "expr.mtx").tocsr().T.tocsr()
    features = pd.read_csv(check_dir / "features.csv", header=None)[0].tolist()
    meta = pd.read_csv(check_dir / "meta.csv")
    lr = pd.read_csv(check_dir / "lr.csv")
    draws = json.loads((check_dir / "subsamples.json").read_text())
    s, reps = next(iter(draws.items()))
    rows = np.flatnonzero(meta["sample"].to_numpy() == s)
    cells = pd.Index(meta["cell"].to_numpy()[rows])
    lig = entity_expression(X[rows], features, lr["ligand"], "min")
    rec = entity_expression(X[rows], features, lr["receptor"], "min")
    subs = [cells.get_indexer(r) for r in reps]
    sim = scriabin.averaged_similarity(lig, rec, lr["lr_name"].to_numpy(), subs)
    return sim.to_numpy()


def write(d: Path, check_dirs: list[Path]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(4).normal(size=(400, 40))
    sims = {
        "flat": 0.5 + 0.5 * np.corrcoef(x, rowvar=False),
        "blocks": _block_similarity(),
    }
    for c in check_dirs:
        dataset = (c / "dataset.txt").read_text().strip()
        sims[f"{dataset}"] = _sample_similarity(c)
    for tag, sim in sims.items():
        np.savetxt(d / f"sim_{tag}.csv", sim, delimiter=",", fmt="%.17g")
    SETTINGS.to_csv(d / "settings.csv", index=False, na_rep="NA")
    print(f"wrote {list(sims)} ({[len(s) for s in sims.values()]} LR pairs)")


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def _load(d: Path, name: str) -> np.ndarray:
    return np.loadtxt(d / name, delimiter=",", ndmin=2)


def _compare_setting(d: Path, tag: str, k: int, sim: np.ndarray) -> None:
    cfg = settings(SETTINGS.iloc[k - 1])
    id_ = f"{tag}_{k}"
    r_mods = pd.read_csv(d / f"r_mods_{id_}.csv")
    power = scriabin.choose_soft_power(sim, cfg)
    check(
        f"{id_} soft power",
        power == r_mods["power"].iloc[0],
        f"py {power} R {r_mods['power'].iloc[0]}",
    )
    adj = sim**power
    tom = _wgcna.signed_tom(adj)
    diff = np.abs(tom - _load(d, f"r_tom_{id_}.csv")).max()
    check(f"{id_} TOM", diff < TOL, f"max |diff| {diff:.2e}")
    tree = _wgcna.hclust_complete(1.0 - tom)
    r_merge = _load(d, f"r_merge_{id_}.csv")
    same_tree = np.array_equal(
        tree.merge, r_merge[:, :2].astype(int)
    ) and np.array_equal(tree.order, _load(d, f"r_order_{id_}.csv").ravel().astype(int))
    hdiff = np.abs(tree.height - r_merge[:, 2]).max()
    check(
        f"{id_} tree",
        same_tree and hdiff < TOL,
        f"merge/order equal {same_tree}; height {hdiff:.2e}",
    )
    labels = _wgcna.cutree_dynamic_tree(
        tree,
        float(np.quantile(tree.height, cfg.cut_height_quantile)),
        cfg.min_module_size,
    )
    same = np.array_equal(labels, r_mods["mods"].to_numpy())
    check(
        f"{id_} modules",
        same,
        f"{labels.max()} modules, {(labels == 0).sum()} unassigned",
    )
    con = _wgcna.intramodular_connectivity(adj, labels)
    cdiff = np.abs(con - r_mods[["kTotal", "kWithin"]].to_numpy()).max()
    check(f"{id_} connectivity", cdiff < 1e-6, f"max |diff| {cdiff:.2e}")


def compare(d: Path) -> None:
    for f in sorted(d.glob("sim_*.csv")):
        tag = f.stem.removeprefix("sim_")
        sim = np.loadtxt(f, delimiter=",")
        _, table = _wgcna.pick_soft_threshold(sim, r2_cut=0.6)
        r_fit = pd.read_csv(d / f"r_fit_{tag}.csv")
        diff = np.abs(table - r_fit[["SFT.R.sq", "slope"]].to_numpy()).max()
        check(f"{tag} scale-free fit table", diff < 1e-6, f"max |diff| {diff:.2e}")
        for k in range(1, len(SETTINGS) + 1):
            _compare_setting(d, tag, k, sim)


if __name__ == "__main__":
    d, stage = Path(sys.argv[1]), sys.argv[2]
    if stage == "write":
        write(d, [Path(p) for p in sys.argv[3:]])
    else:
        compare(d)
        (d / "compare_wgcna.json").write_text(json.dumps(RESULTS, indent=1))
        n_fail = sum(not r["ok"] for r in RESULTS)
        print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks pass")
        sys.exit(1 if n_fail else 0)
