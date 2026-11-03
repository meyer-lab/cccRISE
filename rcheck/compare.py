"""Compare the Python comparators with the upstream R outputs written by
rcheck/run_upstream.R, on the inputs from rcheck/export_inputs.py.

Usage:
  uv run python rcheck/compare.py <dir> programs      # also writes random_sets.json
  uv run python rcheck/compare.py <dir> significance
  uv run python rcheck/compare.py <dir> scores

Prints one line per check (max abs difference or equality) and exits non-zero
if any check fails its tolerance; results are also written to
<dir>/compare_<stage>.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from cccrise.comparators import niches, scriabin
from cccrise.comparators.inputs import entity_expression

N_REPLICATE_CHECK = 200
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def close(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    diff = np.nanmax(np.abs(a - b)) if a.size else 0.0
    check(
        name,
        a.shape == b.shape and same_nan and diff <= tol,
        f"max |diff| {diff:.2e} (tol {tol:.0e}), shape {a.shape}",
    )


def load(d: Path):
    X = scipy.io.mmread(d / "expr.mtx").T.tocsr()  # cells x features
    features = pd.read_csv(d / "features.csv", header=None)[0].astype(str).to_numpy()
    meta = pd.read_csv(d / "meta.csv")
    lr = pd.read_csv(d / "lr.csv")
    return X, pd.Index(features), meta, lr


def programs(d: Path) -> None:
    X, features, meta, lr = load(d)
    draws = json.loads((d / "subsamples.json").read_text())
    names = lr["lr_name"].to_numpy()
    py_modules = {}
    for s, reps in draws.items():
        rows = np.nonzero(meta["sample"].to_numpy() == s)[0]
        cells = pd.Index(meta["cell"].to_numpy()[rows])
        lig = entity_expression(X[rows], features, lr["ligand"], "min")
        rec = entity_expression(X[rows], features, lr["receptor"], "min")
        subs = [cells.get_indexer(r) for r in reps]
        res = scriabin.interaction_programs(
            lig,
            rec,
            names,
            meta["celltype"].to_numpy()[rows],
            np.random.default_rng(0),
            subs,
        )
        r_sim = pd.read_csv(d / f"r_sim_{s}.csv", index_col=0)
        check(
            f"scriabin[{s}] similarity labels",
            list(r_sim.index) == list(res.similarity.index),
            f"{len(r_sim)} LR pairs",
        )
        close(
            f"scriabin[{s}] similarity",
            res.similarity.to_numpy(),
            r_sim.to_numpy(),
            1e-12,
        )
        r_mods = json.loads((d / f"r_modules_{s}.json").read_text())
        check(
            f"scriabin[{s}] modules",
            res.modules == r_mods,
            f"{len(res.modules)} Python / {len(r_mods)} R modules",
        )
        r_con = pd.read_csv(d / f"r_con_{s}.csv", index_col=0)
        close(
            f"scriabin[{s}] kTotal",
            res.connectivity["kTotal"],
            r_con.loc[res.connectivity.index, "kTotal"],
            1e-9,
        )
        close(
            f"scriabin[{s}] kWithin",
            res.connectivity["kWithin"],
            r_con.loc[res.connectivity.index, "kWithin"],
            1e-9,
        )
        py_modules[s] = res.modules
    r_per_sample = {
        s: json.loads((d / f"r_modules_{s}.json").read_text()) for s in draws
    }
    r_merged = json.loads((d / "r_merged.json").read_text())
    check(
        "scriabin merge (R modules in)",
        scriabin.merge_programs(r_per_sample) == r_merged,
        f"{len(r_merged)} merged programs",
    )
    check(
        "scriabin merge (Python modules in)",
        scriabin.merge_programs(py_modules) == r_merged,
        "",
    )

    # Random LR sets for the significance stage, drawn here and fed to R
    rng = np.random.default_rng(7)
    sets = {}
    for s in draws:
        index = pd.read_csv(d / f"r_sim_{s}.csv", index_col=0).index
        # A module with no LR pair in this sample fails upstream's Wilcoxon on
        # its first draw (try() -> 1), so R consumes a single draw for it
        reps = {
            m: N_REPLICATE_CHECK if index.isin(v).any() else 1
            for m, v in r_merged.items()
        }
        sets[s] = [
            [
                rng.choice(len(index), len(v), replace=False).tolist()
                for _ in range(reps[m])
            ]
            for m, v in r_merged.items()
        ]
    (d / "random_sets.json").write_text(json.dumps(sets))

    niches_checks(d, X, features, meta, lr)


def niches_checks(d: Path, X, features, meta, lr) -> None:
    lig_all = entity_expression(X, features, lr["ligand"], "product")
    rec_all = entity_expression(X, features, lr["receptor"], "product")
    for s in meta["sample"].unique():
        in_sample = meta["sample"].to_numpy() == s
        keep = np.zeros(len(meta), bool)
        keep[in_sample] = niches.filter_cells_per_type(
            meta["celltype"].to_numpy()[in_sample], 1
        )
        cells = meta["cell"].to_numpy()[keep]
        lig, rec = lig_all[keep], rec_all[keep]
        for blend in niches.BLENDS:
            for mode, fn in (
                ("s2c", niches.system_to_cell),
                ("c2s", niches.cell_to_system),
            ):
                r = pd.read_csv(d / f"r_{mode}_{s}_{blend}.csv", index_col=0)
                r_cells = [c.split("—")[1 if mode == "s2c" else 0] for c in r.columns]
                order = pd.Index(cells).get_indexer(r_cells)
                check(
                    f"niches[{s}] {mode} {blend} cells",
                    (order >= 0).all() and len(order) == len(cells),
                    f"{len(order)} cells",
                )
                close(
                    f"niches[{s}] {mode} {blend}",
                    fn(lig, rec, blend)[order].T,
                    r.to_numpy(),
                    1e-10,
                )
        r = pd.read_csv(d / f"r_c2c_{s}.csv", index_col=0)
        rmeta = pd.read_csv(d / f"r_c2c_meta_{s}.csv", index_col=0)
        pairs = pd.DataFrame([c.split("—") for c in r.columns], columns=["s", "r"])
        idx = pd.Index(cells)
        edges = pd.DataFrame(
            {
                "sender": idx.get_indexer(pairs["s"]),
                "receiver": idx.get_indexer(pairs["r"]),
            }
        )
        close(
            f"niches[{s}] c2c values on R's pairs",
            niches.cell_to_cell(lig, rec, edges).T,
            r.to_numpy(),
            1e-10,
        )
        types = pd.Series(meta["celltype"].to_numpy()[keep])
        n_type = types.value_counts()
        block = rmeta.groupby(["SendingType", "ReceivingType"]).size()
        expected = {
            (a, b): min(n_type[a], n_type[b])
            for a in n_type.index
            for b in n_type.index
        }
        check(
            f"niches[{s}] c2c block sizes",
            block.to_dict() == expected,
            f"{len(block)} type pairs",
        )
        py = niches.cell_to_cell_pairs(types.to_numpy(), np.random.default_rng(0))
        py_block = py.groupby(["sending_type", "receiving_type"]).size().to_dict()
        check(
            f"niches[{s}] c2c Python block sizes",
            py_block == expected,
            f"{len(py)} edges",
        )
        no_repeat = (
            rmeta.assign(s=pairs["s"].to_numpy(), r=pairs["r"].to_numpy())
            .groupby(["SendingType", "ReceivingType"])
            .agg(lambda x: x.is_unique)
        )
        check(
            f"niches[{s}] c2c no repeats within a block (R)",
            bool(no_repeat.to_numpy().all()),
            "",
        )


def significance(d: Path) -> None:
    sets = json.loads((d / "random_sets.json").read_text())
    r_merged = json.loads((d / "r_merged.json").read_text())
    sims = {s: pd.read_csv(d / f"r_sim_{s}.csv", index_col=0) for s in sets}
    py = pd.DataFrame(index=list(r_merged), columns=list(sets), dtype=float)
    for s, per_mod in sets.items():
        for (name, members), rs in zip(r_merged.items(), per_mod, strict=True):
            py.loc[name, s] = scriabin.module_significance(
                sims[s], members, np.array(rs)
            )
    r = pd.read_csv(d / "r_significance.csv")
    r_tab = r.drop_duplicates("name").set_index("name")[[f"{s}_pval" for s in sets]]
    kept = scriabin.kept_programs(r_merged, py)
    check(
        "scriabin kept programs",
        sorted(kept) == sorted(r_tab.index),
        f"{len(kept)} Python / {len(r_tab)} R kept",
    )
    common = [n for n in r_tab.index if n in py.index]
    close(
        "scriabin significance fractions",
        py.loc[common].to_numpy(),
        r_tab.loc[common].to_numpy(),
        1e-12,
    )


def scores(d: Path) -> None:
    X, features, _, _ = load(d)
    r_merged = json.loads((d / "r_merged.json").read_text())
    ctrl = json.loads((d / "r_controls.json").read_text())
    r = pd.read_csv(d / "r_scores.csv", index_col=0)
    for side, i in (("ligands", 0), ("receptors", 1)):
        for k, (name, members) in enumerate(r_merged.items()):
            feats = scriabin.program_sides(members)[i]
            c = ctrl[side]
            py = scriabin.module_score(
                X, features, feats, c[name] if isinstance(c, dict) else c[k]
            )
            close(
                f"module score {side} {name} (R controls)",
                py,
                r[f"{side}_{name}"].to_numpy(),
                1e-10,
            )
    # Own control draws: AddModuleScore's controls are random, so agreement
    # with R is judged against Python's own seed-to-seed agreement
    runs = [
        scriabin.score_programs(X, features, r_merged, seed=s) for s in range(1, 11)
    ]
    for side, i in (("ligands", 0), ("receptors", 1)):
        vs_r = np.array(
            [
                [np.corrcoef(run[i][n], r[f"{side}_{n}"])[0, 1] for n in r_merged]
                for run in runs
            ]
        )
        vs_py = np.array(
            [
                [np.corrcoef(runs[a][i][n], runs[b][i][n])[0, 1] for n in r_merged]
                for a in range(10)
                for b in range(a + 1, 10)
            ]
        )
        ok = (
            vs_r.min() >= vs_py.min() - 0.02
            and abs(np.median(vs_r) - np.median(vs_py)) <= 0.01
        )
        check(
            f"module score {side} (own controls) within seed-to-seed spread",
            ok,
            f"vs R min/median r {vs_r.min():.3f}/{np.median(vs_r):.3f}; seed vs seed {vs_py.min():.3f}/{np.median(vs_py):.3f}",
        )
    means = pd.Series(np.asarray(X.mean(axis=0)).ravel(), index=features)
    bins = scriabin.expression_bins(means, np.random.default_rng(1))
    check("expression bins", bins.nunique() == scriabin.N_BIN, f"{bins.nunique()} bins")


if __name__ == "__main__":
    d, stage = Path(sys.argv[1]), sys.argv[2]
    {"programs": programs, "significance": significance, "scores": scores}[stage](d)
    (d / f"compare_{stage}.json").write_text(json.dumps(RESULTS, indent=1))
    sys.exit(0 if all(r["ok"] for r in RESULTS) else 1)
