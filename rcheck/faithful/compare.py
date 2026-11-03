"""Tables for the faithful-Scriabin report (`SCRIABIN_FAITHFUL_REPORT.md`):
stability of upstream Scriabin run as its authors set it up (`ip_runs.R`,
`merge_sig.R`) under seeds, cell removal and UMI downsampling; the paper's
Extended Data Fig. 8f quantities on ALAD; and agreement between that run and
the vendored port on cccRISE's inputs (programs, per-cell scores, cell-type
pair weights). Writes CSVs to ``rcheck/out/faithful/report/`` and prints them.

Programs are compared as sets of (ligand gene, receptor gene) pairs: the
port's LR pairs carry complexes (``WNT2B=FZD1&LRP5``) where OmniPath lists
subunits, so each pair is expanded to its gene pairs, and port-vs-faithful
comparisons are restricted to gene pairs present in both universes.

    uv run python rcheck/faithful/compare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cccrise.validation.singlecell_joint import module_stability
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

FAITHFUL = Path("rcheck/out/faithful")
RUNS = FAITHFUL / "runs"
VENDORED = Path("singlecell_cache/alad")
OUT = FAITHFUL / "report"
MAIN = "alra_s1111"
N_NULL = 200

Modules = dict[str, set[tuple[str, str]]]


def gene_pairs(members: list[str]) -> set[tuple[str, str]]:
    out = set()
    for m in members:
        lig, rec = m.split("=")
        out |= {(a, b) for a in lig.split("&") for b in rec.split("&")}
    return out


def as_modules(raw: dict) -> Modules:
    return {
        k: gene_pairs(v if isinstance(v, list) else [v]) for k, v in raw.items() if v
    }


def stability(a: Modules, b: Modules) -> float:
    """`singlecell_joint.module_stability` on gene-pair sets; NaN if either
    side is empty."""
    if not a or not b:
        return np.nan
    fmt = {k: [f"{x}={y}" for x, y in v] for k, v in a.items()}
    fmt_b = {k: [f"{x}={y}" for x, y in v] for k, v in b.items()}
    return module_stability(fmt, fmt_b)


def restrict(mods: Modules, universe: set, min_size: int = 2) -> Modules:
    out = {k: v & universe for k, v in mods.items()}
    return {k: v for k, v in out.items() if len(v) >= min_size}


def null_stability(a: Modules, b: Modules, universe: set, rng) -> float:
    """Mean stability of `b` against random modules with `a`'s sizes."""
    pool = sorted(universe)
    vals = []
    for _ in range(N_NULL):
        rand = {
            k: {pool[i] for i in rng.choice(len(pool), len(v), replace=False)}
            for k, v in a.items()
        }
        vals.append(stability(rand, b))
    return float(np.mean(vals))


# --- loading ---------------------------------------------------------------------


def run_samples(run: str) -> dict[str, dict]:
    d = RUNS / run
    return {
        p.stem: json.loads(p.read_text())
        for p in sorted(d.glob("*.json"))
        if p.stem not in ("merged", "kept")
    }


def run_programs(run: str, which: str) -> Modules | None:
    p = RUNS / run / f"{which}.json"
    return as_modules(json.loads(p.read_text())) if p.exists() else None


def vendored_samples(folder: Path) -> dict[str, Modules]:
    return {
        p.stem: as_modules(json.loads(p.read_text())["modules"])
        for p in sorted(folder.glob("*.json"))
    }


# --- tables ----------------------------------------------------------------------


def per_sample_stability(a: str, b: str) -> pd.DataFrame:
    sa, sb = run_samples(a), run_samples(b)
    rows = []
    for s in sorted(set(sa) & set(sb)):
        rows.append(
            {
                "sample": s,
                "n_cells": sa[s]["n_cells"],
                "stability": stability(
                    as_modules(sa[s]["modules"]), as_modules(sb[s]["modules"])
                ),
            }
        )
    return pd.DataFrame(rows).assign(a=a, b=b)


def perturbation_table(pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Per perturbation: per-sample module stability (median, IQR, over
    subsampled samples and samples of <= 300 cells) and merged / kept
    program stability."""
    rows = []
    for label, a, b in pairs:
        if not (RUNS / a).exists() or not (RUNS / b).exists():
            continue
        ps = per_sample_stability(a, b)
        sub = ps[ps["n_cells"] > 300]["stability"]
        small = ps[ps["n_cells"] <= 300]["stability"]
        row = {
            "perturbation": label,
            "samples": len(ps),
            "per_sample_median": sub.median(),
            "per_sample_q25": sub.quantile(0.25),
            "per_sample_q75": sub.quantile(0.75),
            "per_sample_small_median": small.median(),
        }
        for which in ("merged", "kept"):
            pa, pb = run_programs(a, which), run_programs(b, which)
            row[f"{which}_stability"] = stability(pa, pb) if pa and pb else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ed8f_table(run: str) -> pd.DataFrame:
    """Extended Data Fig. 8f quantities for one run: per sample the soft power
    WGCNA recommends at R^2 0.6 (upstream falls back to 1 when it is NA or
    > 3); per merged program whether it failed significance in every sample,
    and whether it has a single ligand or receptor; program sizes."""
    samples = run_samples(run)
    est = pd.Series({s: v["power_estimate"] for s, v in samples.items()})
    merged = json.loads((RUNS / run / "merged.json").read_text())
    kept = set(json.loads((RUNS / run / "kept.json").read_text()))
    sides = {
        k: ({m.split("=")[0] for m in v}, {m.split("=")[1] for m in v})
        for k, v in merged.items()
    }
    return pd.DataFrame(
        [
            {
                "run": run,
                "samples": len(est),
                "power_estimate_median": est.median(),
                "power_fallback_to_1": int((est.isna() | (est > 3)).sum()),
                "merged_programs": len(merged),
                "failed_significance_pct": 100
                * np.mean([k not in kept for k in merged]),
                "single_ligand_or_receptor_pct": 100
                * np.mean([min(len(l), len(r)) == 1 for l, r in sides.values()]),
                "mean_ligands": np.mean([len(l) for l, _ in sides.values()]),
                "mean_receptors": np.mean([len(r) for _, r in sides.values()]),
                "kept_programs": len(kept),
            }
        ]
    )


def lr_universe() -> set[tuple[str, str]]:
    """(ligand, receptor) gene pairs both versions can use: OmniPath pairs
    with both genes measured, and the port's LR pairs expanded to subunits."""
    from cccrise.comparators.inputs import fit_lr_pairs
    from cccrise.factors import load_factors_netcdf

    used = pd.read_csv(FAITHFUL / "depth1" / "lr_used.csv")
    faithful = set(zip(used.source_genesymbol, used.target_genesymbol, strict=True))
    lr = fit_lr_pairs(load_factors_netcdf("cccrise/data/factors/bal_alad_factors.nc"))
    return faithful & gene_pairs(list(lr["lr_name"]))


def faithful_vs_vendored(main: str = MAIN, retest: str = "alra_s2") -> pd.DataFrame:
    """Per-sample modules and merged / kept programs, faithful vs port,
    restricted to the gene pairs both can use (`lr_universe`); with the
    faithful run's own seed-to-seed agreement on the same pairs (the
    ceiling) and a size-matched random null."""
    rng = np.random.default_rng(0)
    universe = lr_universe()
    vend = vendored_samples(VENDORED / "scriabin" / "programs")
    faith = {s: as_modules(v["modules"]) for s, v in run_samples(main).items()}
    again = {s: as_modules(v["modules"]) for s, v in run_samples(retest).items()}
    rows = []
    for s in sorted(set(vend) & set(faith)):
        f, v = restrict(faith[s], universe), restrict(vend[s], universe)
        r = restrict(again.get(s, {}), universe)
        rows.append(
            {
                "level": "per-sample modules",
                "sample": s,
                "faithful_vs_port": stability(f, v),
                "faithful_vs_faithful_seed": stability(f, r) if r else np.nan,
                "null": null_stability(f, v, universe, rng) if f and v else np.nan,
            }
        )
    for which, vfile in (("merged", "merged_programs"), ("kept", "kept_programs")):
        f = run_programs(main, which)
        r = run_programs(retest, which)
        v = as_modules(
            json.loads((VENDORED / "scriabin" / f"{vfile}.json").read_text())
        )
        if f is None:
            continue
        f, v = restrict(f, universe), restrict(v, universe)
        r = restrict(r, universe) if r else None
        rows.append(
            {
                "level": f"{which} programs",
                "sample": "all",
                "faithful_vs_port": stability(f, v),
                "faithful_vs_faithful_seed": stability(f, r) if r else np.nan,
                "null": null_stability(f, v, universe, rng),
            }
        )
    return pd.DataFrame(rows).assign(faithful=main)


def _match(f: Modules, v: Modules) -> list[tuple[str, str, float]]:
    """Hungarian matching of `f`'s programs to `v`'s on Jaccard."""
    fk, vk = list(f), list(v)
    jac = np.array(
        [[len(f[a] & v[b]) / max(len(f[a] | v[b]), 1) for b in vk] for a in fk]
    )
    rows, cols = linear_sum_assignment(-jac)
    return [(fk[i], vk[j], jac[i, j]) for i, j in zip(rows, cols, strict=True)]


def _type_pair_weights(lig: pd.Series, rec: pd.Series, types: pd.Series) -> np.ndarray:
    """IPCellTypeSummary's additive score for every sender x receiver type:
    mean ligand score of the sender type + mean receptor score of the
    receiver type."""
    s = lig.groupby(types).mean()
    r = rec.groupby(types).mean()
    return (s.to_numpy()[:, None] + r.to_numpy()[None, :]).ravel()


def faithful_source(run: str) -> tuple[Modules, pd.DataFrame, pd.DataFrame] | None:
    d = RUNS / run
    if not (d / "scores_ligands.csv").exists():
        return None
    return (
        run_programs(run, "kept"),
        pd.read_csv(d / "scores_ligands.csv", index_col=0),
        pd.read_csv(d / "scores_receptors.csv", index_col=0),
    )


def port_source(folder: str) -> tuple[Modules, pd.DataFrame, pd.DataFrame]:
    d = VENDORED / folder
    return (
        as_modules(json.loads((d / "kept_programs.json").read_text())),
        pd.read_parquet(d / "scores_sender.parquet"),
        pd.read_parquet(d / "scores_receiver.parquet"),
    )


def score_agreement(
    a, b, types: pd.Series, universe: set | None = None
) -> pd.DataFrame:
    """Kept programs of `a` matched to `b`'s (Hungarian on Jaccard, over
    `universe` when given): the Jaccard, Spearman of per-cell sender and
    receiver scores over all cells, and Pearson of the sender x receiver
    cell-type weights (`IPCellTypeSummary`'s additive score)."""
    (fa, la, ra), (fb, lb, rb) = a, b
    if universe is not None:
        fa, fb = restrict(fa, universe, 1), restrict(fb, universe, 1)
    cells = la.index.intersection(lb.index)
    t = types.loc[cells]
    rows = []
    for x, y, jac in _match(fa, fb):
        sa, sb = la.loc[cells, x], lb.loc[cells, y]
        qa, qb = ra.loc[cells, x], rb.loc[cells, y]
        rows.append(
            {
                "program": x,
                "match": y,
                "jaccard": jac,
                "sender_spearman": spearmanr(sa, sb)[0],
                "receiver_spearman": spearmanr(qa, qb)[0],
                "type_pair_pearson": np.corrcoef(
                    _type_pair_weights(sa, qa, t), _type_pair_weights(sb, qb, t)
                )[0, 1],
            }
        )
    return pd.DataFrame(rows)


def score_table(types: pd.Series) -> pd.DataFrame:
    """`score_agreement` for faithful vs port (shared gene pairs), and each
    version against itself rerun with another seed (all its pairs)."""
    tables = []
    main, again = faithful_source(MAIN), faithful_source("alra_s2")
    port, port_again = port_source("scriabin"), port_source("t3a_retest/scriabin")
    if main is not None:
        tables.append(
            score_agreement(main, port, types, lr_universe()).assign(
                comparison="faithful vs port"
            )
        )
    if main is not None and again is not None:
        tables.append(
            score_agreement(main, again, types).assign(
                comparison="faithful vs faithful, seed 2"
            )
        )
    tables.append(
        score_agreement(port, port_again, types).assign(
            comparison="port vs port, new seed"
        )
    )
    return pd.concat(tables)


def _benchmark_programs(source, samples: pd.Series):
    """A `singlecell_benchmark.Programs` from kept programs and scores, with
    LR pairs named ``ligand=receptor`` over every gene pair it uses."""
    from cccrise.validation import singlecell_benchmark as sb

    mods, lig, rec = source
    names = sorted({f"{a}={b}" for v in mods.values() for a, b in v})
    kept = {k: [f"{a}={b}" for a, b in v] for k, v in mods.items()}
    lr = pd.DataFrame({"lr_name": names})
    return sb.scriabin_programs(kept, lr, lig, rec, samples)


def retention_table(samples: pd.Series) -> pd.DataFrame:
    """The benchmark's T3a-retest metric (`singlecell_joint.retention`:
    Hungarian matching on membership cosine >= 0.5, then the mean of sender
    and receiver Spearman; 0 when unmatched) for each version rerun with a
    new seed. The port's row recomputes the benchmark's published number."""
    from cccrise.validation import singlecell_joint as sj

    pairs = [
        ("port, new seed", port_source("scriabin"), port_source("t3a_retest/scriabin")),
        ("faithful, seed 2", faithful_source(MAIN), faithful_source("alra_s2")),
    ]
    rows = []
    for label, a, b in pairs:
        if a is None or b is None:
            continue
        pa, pb = _benchmark_programs(a, samples), _benchmark_programs(b, samples)
        # one LR index for both, so membership cosines are comparable
        index = pa.lr_weights.index.union(pb.lr_weights.index)
        la = (pa.lr_weights.reindex(index, fill_value=0) > 0).astype(float)
        lb = (pb.lr_weights.reindex(index, fill_value=0) > 0).astype(float)
        t = sj.retention(pa, pb, la, lb)
        rows.append(
            {
                "comparison": label,
                "programs": len(t),
                "matched": int(t["match"].notna().sum()),
                "retention_mean": t["retention"].mean(),
                "retention_median": t["retention"].median(),
                "retention_matched_mean": t.loc[t["match"].notna(), "retention"].mean(),
            }
        )
    return pd.DataFrame(rows)


CCIM_SAMPLES = (
    "ALAD1-HS58-E1",
    "ALAD1-HS16-E1",
    "ALAD1-HS62-E1",
    "ALAD1-HS18-E1",
    "ALAD1-HS40-E1",
    "ALAD1-HS1-E1",
)


def _faithful_lr_expression(kind: str):
    import scipy.io

    d = FAITHFUL / "depth1"
    X = scipy.io.mmread(d / f"{kind}_lr.mtx").T.tocsr()  # cells x genes
    genes = pd.Index((d / f"{kind}_lr_genes.txt").read_text().split())
    cells = pd.Index((d / f"{kind}_lr_cells.txt").read_text().split())
    return X, genes, cells


def ccim_weights() -> pd.DataFrame:
    """Which cell pairs each version weights: per sample, the summarized
    interaction ``S[s, r] = sum_l sqrt(L[s, l] * R[r, l])`` (the CCIM summed
    over LR pairs; Scriabin's summarized interaction graph) over the simple
    LR pairs both versions contain, faithful (SCT or ALRA) against the port's
    inputs: Spearman over all cell pairs, and over sender and receiver totals."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _common import dataset_inputs

    inp = dataset_inputs("alad")
    used = pd.read_csv(FAITHFUL / "depth1" / "lr_used.csv")
    faithful_pairs = set(
        zip(used.source_genesymbol, used.target_genesymbol, strict=True)
    )
    names = inp.lr["lr_name"].to_numpy()
    shared = [
        i
        for i, n in enumerate(names)
        if "&" not in n and tuple(n.split("=")) in faithful_pairs
    ]
    lig_genes = [names[i].split("=")[0] for i in shared]
    rec_genes = [names[i].split("=")[1] for i in shared]
    rows = []
    for kind in ("sct", "alra"):
        X, genes, cells = _faithful_lr_expression(kind)
        pos = cells.get_indexer(inp.obs.index)
        for s in CCIM_SAMPLES:
            idx = np.flatnonzero(inp.obs["sample"].to_numpy() == s)
            fx = X[pos[idx]]
            fl = np.sqrt(fx[:, genes.get_indexer(lig_genes)].toarray())
            fr = np.sqrt(fx[:, genes.get_indexer(rec_genes)].toarray())
            lig, rec = inp.lr_expression(idx)
            pl, pr = np.sqrt(lig[:, shared]), np.sqrt(rec[:, shared])
            sf, sp = fl @ fr.T, pl @ pr.T
            rows.append(
                {
                    "faithful": kind,
                    "sample": s,
                    "n_cells": len(idx),
                    "shared_lr_pairs": len(shared),
                    "cell_pair_spearman": spearmanr(sf.ravel(), sp.ravel())[0],
                    "sender_total_spearman": spearmanr(sf.sum(1), sp.sum(1))[0],
                    "receiver_total_spearman": spearmanr(sf.sum(0), sp.sum(0))[0],
                }
            )
    return pd.DataFrame(rows)


def cell_drop_table() -> pd.DataFrame:
    d = RUNS / "drop"
    rows = []
    for base in sorted(d.glob("*_drop-1.json")):
        s = base.name.removesuffix("_drop-1.json")
        m0 = as_modules(json.loads(base.read_text())["modules"])
        for p in sorted(d.glob(f"{s}_drop*.json")):
            if p == base:
                continue
            rows.append(
                {
                    "sample": s,
                    "dropped": p.stem.split("_drop")[1],
                    "stability": stability(
                        m0, as_modules(json.loads(p.read_text())["modules"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    fmt = {"float_format": lambda x: f"{x:.3f}", "index": False}
    fmt_g = {"float_format": lambda x: f"{x:.3f}"}
    seeds = [
        ("ALRA, seed 1111 vs 2", MAIN, "alra_s2"),
        ("ALRA, seed 1111 vs 3", MAIN, "alra_s3"),
        ("ALRA, seed 2 vs 3", "alra_s2", "alra_s3"),
        ("SCT only, seed 1111 vs 2", "sct_s1111", "sct_s2"),
        ("ALRA, 50% of cells", MAIN, "alra_f50_s1111"),
        # HS68 errors in upstream at x0.52 / x0.25: compared on the other 46
        ("ALRA, UMIs to inDrop depth (x0.52)", "alra_s1111_no68", "alra_d0.52_s1111"),
        ("ALRA, UMIs x0.25", "alra_s1111_no68", "alra_d0.25_s1111"),
        ("ALRA, UMIs x0.10", MAIN, "alra_d0.1_s1111"),
        ("ALRA vs SCT only (seed 1111)", MAIN, "sct_s1111"),
    ]
    tables = {
        "perturbations": perturbation_table(seeds),
        "ed8f": pd.concat(
            [
                ed8f_table(r)
                for r in (MAIN, "sct_s1111")
                if (RUNS / r / "kept.json").exists()
            ]
        ),
        "cell_drop": cell_drop_table(),
    }
    tables["faithful_vs_port"] = pd.concat(
        [
            faithful_vs_vendored(MAIN, "alra_s2"),
            faithful_vs_vendored("sct_s1111", "sct_s2"),
        ]
    )
    meta = pd.read_csv(FAITHFUL / "input" / "meta.csv", index_col=0)
    tables["scores"] = score_table(meta["celltype"])
    tables["retention"] = retention_table(meta["sample"])
    tables["ccim_weights"] = ccim_weights()
    lisi = sorted((FAITHFUL / "lisi").glob("*.csv"))
    if lisi:
        tables["lisi"] = pd.concat([pd.read_csv(p) for p in lisi])
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n## {name}")
        if name == "faithful_vs_port":
            cols = ["faithful_vs_port", "faithful_vs_faithful_seed", "null"]
            print(t.groupby(["faithful", "level"])[cols].median().to_string(**fmt_g))
        elif name == "scores":
            cols = [
                "jaccard",
                "sender_spearman",
                "receiver_spearman",
                "type_pair_pearson",
            ]
            print(t.groupby("comparison")[cols].median().to_string(**fmt_g))
        else:
            print(t.to_string(**fmt))


if __name__ == "__main__":
    main()
