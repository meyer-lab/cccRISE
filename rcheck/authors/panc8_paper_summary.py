"""Summaries of the Scriabin paper's robustness analyses reproduced on panc8
(rcheck/authors/panc8_paper_{r2_sweep,noise}.R; jobs in
rcheck/out/authors/panc8_paper/jobs.tsv, run by rcheck/shared/loadq.py):

- ``r2``: Extended Data Fig. 8f's quantities per subsample size, R^2 and
  power rule (the paper's: recommended power, % programs failing
  significance, % programs of one ligand or receptor, program size), and
  the read-out the paper does not make: best-match Jaccard of the programs
  between every pair of subsamples.
- ``noise``: Fig. 2e's LISI read-out per subsample, and best-match Jaccard of
  the programs: ground truth vs downsampled, and between the two independent
  downsampling draws of the same cells.

  uv run python rcheck/authors/panc8_paper_summary.py
Writes r2_by_subsample.csv, r2_summary.csv, noise_lisi.csv,
noise_programs.csv under rcheck/out/authors/panc8_paper/. The tables the
figure reads (figureR_scriabin_reproduced) are the runner's:
`uv run cccrise-scriabin-reproduction`.
"""

import itertools

import numpy as np
import pandas as pd

from cccrise.runners.run_scriabin_reproduction import (
    RUNS_DIR,
    composition,
    load_runs,
    program_pairs,
)
from cccrise.runners.run_scriabin_stability import best_match

ROOT = RUNS_DIR
_load = load_runs
_composition = composition


def _sets(modules: dict) -> list[set]:
    return [set(p) for p in program_pairs(modules)]


def r2() -> None:
    runs = _load(ROOT / "r2_sweep")
    rows = []
    for name, run in runs.items():
        for p in run["powers"]:
            for rule in ("uncapped", "capped"):
                key = str(int(p[rule]))
                comp = _composition(run["modules"][key])
                fail = np.asarray(run["failure"][key], float)
                rows.append(
                    {
                        "n_cells": run["n_cells"],
                        "subsample": run["index"],
                        "run": name,
                        "r2": p["r2"],
                        "rule": rule,
                        "recommended": p["recommended"],
                        "power": int(p[rule]),
                        "programs": len(comp),
                        "pct_failing": 100 * float(np.mean(fail > 0.05)),
                        "pct_single_ligand_or_receptor": 100
                        * float(np.mean((comp.ligands == 1) | (comp.receptors == 1))),
                        "mean_pairs": comp.pairs.mean(),
                        "mean_ligands_receptors": (
                            (comp.ligands + comp.receptors) / 2
                        ).mean(),
                    }
                )
    by_sub = pd.DataFrame(rows)
    by_sub.to_csv(ROOT / "r2_by_subsample.csv", index=False)
    agree = []
    for (n, r2_, rule), g in by_sub.groupby(["n_cells", "r2", "rule"]):
        sets = {
            r.run: _sets(runs[r.run]["modules"][str(r.power)]) for r in g.itertuples()
        }
        vals = [
            best_match(sets[a], sets[b])
            for a, b in itertools.combinations(sorted(sets), 2)
        ]
        q = np.percentile(vals, [25, 50, 75]) if vals else [np.nan] * 3
        agree.append(
            {
                "n_cells": n,
                "r2": r2_,
                "rule": rule,
                "between_subsample_agreement_median": q[1],
                "between_subsample_agreement_iqr": f"{q[0]:.2f}-{q[2]:.2f}",
            }
        )
    summary = (
        by_sub.groupby(["n_cells", "r2", "rule"])
        .agg(
            subsamples=("run", "size"),
            recommended_mean=("recommended", "mean"),
            recommended_na=("recommended", lambda s: int(s.isna().sum())),
            power_mean=("power", "mean"),
            programs=("programs", "mean"),
            pct_failing=("pct_failing", "mean"),
            pct_single=("pct_single_ligand_or_receptor", "mean"),
            mean_ligands_receptors=("mean_ligands_receptors", "mean"),
            mean_pairs=("mean_pairs", "mean"),
        )
        .reset_index()
        .merge(pd.DataFrame(agree), on=["n_cells", "r2", "rule"])
    )
    summary.to_csv(ROOT / "r2_summary.csv", index=False)
    with pd.option_context(
        "display.width", 250, "display.max_columns", 30, "display.precision", 2
    ):
        print(summary)
        for n, g in by_sub[by_sub.rule == "uncapped"].groupby("n_cells"):
            print(
                f"n={n}: Pearson r with R^2 (the paper's panel, uncapped):",
                {
                    c: round(g.r2.corr(g[c]), 2)
                    for c in (
                        "recommended",
                        "pct_failing",
                        "pct_single_ligand_or_receptor",
                        "mean_ligands_receptors",
                    )
                },
            )


NOISE_PAIRS = {  # (a, b): what differs
    ("gt", "down1"): "GT vs downsampled (raw)",
    ("gt", "down1_alra"): "GT vs downsampled + ALRA",
    ("gt_alra", "down1_alra"): "GT + ALRA vs downsampled + ALRA",
    ("down1", "down2"): "two downsampling draws (raw)",
    ("down1_alra", "down2_alra"): "two downsampling draws (ALRA)",
    ("gt", "gt_alra"): "GT, with vs without ALRA",
}


def noise() -> None:
    runs = _load(ROOT / "noise")
    lisi = pd.concat(
        [pd.DataFrame(r["lisi"]) for r in runs.values()], ignore_index=True
    )
    lisi.to_csv(ROOT / "noise_lisi.csv", index=False)
    rows = []
    for run in runs.values():
        progs = run["programs"]
        for (a, b), label in NOISE_PAIRS.items():
            for kind in ("modules", "modules_uncapped"):
                sa, sb = _sets(progs[a][kind]), _sets(progs[b][kind])
                # also on the LR pairs both runs place in some program
                shared = set().union(*sa) & set().union(*sb) if sa and sb else set()
                ra = [s & shared for s in sa if s & shared]
                rb = [s & shared for s in sb if s & shared]
                rows.append(
                    {
                        "subsample": run["index"],
                        "comparison": label,
                        "power_rule": kind,
                        "programs_a": len(sa),
                        "programs_b": len(sb),
                        "agreement": best_match(sa, sb),
                        "agreement_shared_pairs": best_match(ra, rb)
                        if ra and rb
                        else np.nan,
                    }
                )
    # cell resampling, at full depth: GT programs of different 300-cell subsamples
    for a, b in itertools.combinations(sorted(runs), 2):
        sa, sb = (_sets(runs[x]["programs"]["gt"]["modules"]) for x in (a, b))
        rows.append(
            {
                "subsample": f"{a}|{b}",
                "comparison": "GT, two cell subsamples",
                "power_rule": "modules",
                "agreement": best_match(sa, sb),
            }
        )
    prog = pd.DataFrame(rows)
    prog.to_csv(ROOT / "noise_programs.csv", index=False)
    with pd.option_context("display.width", 250, "display.precision", 3):
        print(
            lisi.groupby("arm")[
                ["frac_lisi_gt_1", "lisi_mean", "frac_lisi_gt_1.5", "ccim_spearman"]
            ].agg(["median", "min", "max"])
        )
        print(
            prog.groupby(["power_rule", "comparison"])[
                ["programs_a", "programs_b", "agreement", "agreement_shared_pairs"]
            ].median()
        )
        print(
            {
                k: [round(x, 1) for x in runs[k]["downsampled_mean_counts"]]
                for k in list(runs)[:3]
            }
        )
        print(
            "LR pairs per arm (sub01):",
            {k: v["n_lr"] for k, v in runs["sub01"]["programs"].items()},
        )


if __name__ == "__main__":
    r2()
    noise()
