#!/usr/bin/env python
"""
Runner: the comparator programs cccRISE does not contain, and what they are
(criteria fixed 2026-10-20, before computing).

Scope: every LR column of an own-rank comparator fit (``cccrise-benchmark``'s
cached fits, coarse and fine annotations) whose containment in cccRISE's LR
span is below `CONTAINED` (``cccrise-comparator-agreement``). Each column's
sample, sender and receiver profiles are its own CP factors, or, for a
Tucker (STACCato) LR column, the leading rank-1 HOSVD vectors of its core
slice mapped through the factors (`column_profiles`). Each column is
classed, first match winning (`classify`):

1. one sample: effective samples (participation ratio) below 1.5;
2. marker or homophilic pair: its top pair is homophilic, or both genes
   (every subunit of a complex) are markers of its top sender or top
   receiver type (mean expression >= 4x the rest's, detected in >= 25% of
   the type's cells);
3. abundance readout: |Spearman rho| >= 0.7 between its sample profile and
   the per-sample fraction of its top sender or top receiver type;
4. candidate biology.

Also per column: its 6-month outcome eta^2 and nominal one-way ANOVA p on
read-depth-corrected profiles, and the smallest cccRISE CP rank whose LR span
contains it (production fits of `run_cpd_rank_rule`'s checkpoints, on the
canonical RISE fit and on any ``--rise-factors``).

Writes ``data/validation_results/comparator_audit_<dataset>.parquet``.

    uv run python -m cccrise.runners.run_comparator_audit --dataset alad \\
        --rise-factors output/cache/story/factors/alad_rise34.nc output/cache/story/factors/alad_rise51.nc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..anova import run_multiple_factors_anova
from ..benchmark import ComparatorFit, read_fits
from ..comparators.agreement import CONTAINED, containment_table
from ..factors import load_factors_netcdf
from ..figures.associations import ALAD_ASSOCIATIONS
from ..import_data import add_cond_idxs
from ..rank_selection import screen_starts
from ..utils import correct_conditions
from ..validation.program_breadth import participation_ratio
from ._cli import add_dataset_argument, dataset_metadata, resolve_dataset
from .datasets import VALIDATION_DIR, DatasetSpec
from .provenance import build_metadata, sha256_file, write_metadata
from .run_cpd_rank_rule import CHECKPOINT_DIR, rise_digest

ANNOTATIONS = {"coarse": ("", "broad_cell_type"), "fine": ("_fine", "celltype")}
OUTCOMES = ("6-Month Outcome (4-Way)", "6-Month CLAD/Died vs. No CLAD")
ONE_SAMPLE = 1.5
MARKER_FOLD = 4.0
MARKER_DETECTED = 0.25
ABUNDANCE_RHO = 0.7
COMPLEX_SEP = "&"
N_NULL = 200
HEADER = """\
# Comparator LR columns not contained in cccRISE's span, one row each: their
# pre-registered class, outcome association and the cccRISE rank that
# contains them.
"""


def _unit_sign(vector: np.ndarray) -> np.ndarray:
    """`vector` signed so its largest-magnitude entry is positive."""
    return vector * np.sign(vector[np.argmax(np.abs(vector))])


def column_profiles(fit: ComparatorFit, j: int) -> dict[str, np.ndarray]:
    """LR column `j`'s sample, sender and receiver profiles."""
    if fit.core is None:
        return {"sample": fit.condition_factor[:, j], "sender": fit.modes["sender"][:, j],
                "receiver": fit.modes["receiver"][:, j]}  # fmt: skip
    core = fit.core[:, j, :, :]  # (sample, sender, receiver), STACCato's order
    leading = [
        np.linalg.svd(np.moveaxis(core, mode, 0).reshape(core.shape[mode], -1))[0][:, 0]
        for mode in range(3)
    ]
    factors = (fit.condition_factor, fit.modes["sender"], fit.modes["receiver"])
    return {name: _unit_sign(f @ u) for name, f, u in
            zip(("sample", "sender", "receiver"), factors, leading, strict=True)}  # fmt: skip


def marker_table(adata, column: str, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per type and gene: mean expression over the rest's (fold) and the
    share of the type's cells detecting it."""
    genes = [g for g in genes if g in adata.var_names]
    types = adata.obs[column].astype(str).to_numpy()
    X = adata[:, genes].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    fold, detected = {}, {}
    for t in np.unique(types):
        inside = types == t
        fold[t] = X[inside].mean(0) / np.maximum(X[~inside].mean(0), 1e-12)
        detected[t] = (X[inside] > 0).mean(0)
    return pd.DataFrame(fold, index=genes).T, pd.DataFrame(detected, index=genes).T


def _is_marker(side: str, celltype: str, fold: pd.DataFrame, detected: pd.DataFrame) -> bool:
    genes = side.split(COMPLEX_SEP)
    return all(
        g in fold.columns
        and fold.loc[celltype, g] >= MARKER_FOLD
        and detected.loc[celltype, g] >= MARKER_DETECTED
        for g in genes
    )


def classify(row: dict, fold, detected, fractions: pd.DataFrame, sample) -> str:
    """The pre-registered class of one column (`row` holds its top pair and
    top types; `sample` its sample profile)."""
    if row["effective_samples"] < ONE_SAMPLE:
        return "one sample"
    ligand, receptor = row["top_pair"].split("^")
    if ligand == receptor or any(
        _is_marker(ligand, t, fold, detected) and _is_marker(receptor, t, fold, detected)
        for t in (row["top_sender"], row["top_receiver"])
    ):
        return "marker or homophilic pair"
    rhos = [abs(spearmanr(sample, fractions[t]).statistic)
            for t in (row["top_sender"], row["top_receiver"])]  # fmt: skip
    return "abundance readout" if max(rhos) >= ABUNDANCE_RHO else "candidate biology"


def missed_columns(spec: DatasetSpec, annotation: str) -> pd.DataFrame:
    suffix = ANNOTATIONS[annotation][0]
    table = pd.read_parquet(VALIDATION_DIR / f"comparator_containment_{spec.output_key}{suffix}.parquet")
    table = table[(table["direction"] == "comparator in cccRISE") & (table["mode"] == "lr")
                  & (table["rank_mode"] == "own")]  # fmt: skip
    return table[table["containment"] < CONTAINED]


def outcome_stats(adata, spec: DatasetSpec, samples: np.ndarray) -> pd.DataFrame:
    """Per column (of `samples`, conditions x columns): eta^2 and nominal p
    for each outcome, on read-depth-corrected profiles."""
    adata.uns["A"] = samples
    adata.uns["A"] = correct_conditions(adata)
    configs = [a for a in ALAD_ASSOCIATIONS if a["name"] in OUTCOMES]
    table = run_multiple_factors_anova(adata, associations=configs,
                                       condition_column=spec.condition_column)  # fmt: skip
    return table.pivot_table(index="Component", columns="Association", values=["EtaSquared", "P"])


def production_span(fits_dir: Path, rank: int) -> np.ndarray:
    """The LR factor of `rank`'s best passing production start (best start
    where none passes), as in S3b."""
    starts = []
    for i in range(10):
        z = np.load(fits_dir / f"r{rank}_svd{i}.npz")
        starts.append((z["weights"], [z[f"f{m}"] for m in range(4)], float(z["r2x"])))
    table = screen_starts(starts)
    pool = table[table["Passes"]] if table["Passes"].any() else table
    return starts[int(pool["R2X"].idxmax())][1][3]


def reach(columns: np.ndarray, pairs: np.ndarray, factors: Path, key: str, rng) -> np.ndarray:
    """Per column, the smallest CP rank (1-16) on the RISE fit in `factors`
    whose LR span contains it; NaN if none."""
    order = pd.Index(load_factors_netcdf(factors).coords["lr_pair"].values.astype(str))
    rows = order.get_indexer(pairs)
    fits_dir = CHECKPOINT_DIR / f"{key}_full_{rise_digest(factors)}"
    out = np.full(columns.shape[1], np.nan)
    for rank in range(16, 0, -1):
        span = production_span(fits_dir, rank)
        aligned = np.where(rows[:, None] >= 0, span[np.maximum(rows, 0)], 0.0)
        contained = containment_table(aligned, columns, N_NULL, rng)["contained"].to_numpy()
        out[contained] = rank
    return out


def _celltype_fractions(adata, spec: DatasetSpec, column: str) -> pd.DataFrame:
    counts = pd.crosstab(adata.obs[spec.condition_column].astype(str), adata.obs[column].astype(str))
    return counts.div(counts.sum(axis=1), axis=0)


def _column_record(item, fit: ComparatorFit, pairs: np.ndarray, celltypes: list[str]):
    """One missed column's row, sample profile and LR column."""
    j = int(item.component) - 1
    profiles = column_profiles(fit, j)
    lr = fit.modes["lr"][:, j]
    row = {"annotation": None, "method": item.method, "rank": fit.rank, "component": j + 1,
           "containment": item.containment, "top_pair": pairs[np.argmax(lr**2)],
           "top_share": float(np.max(lr**2) / np.sum(lr**2)),
           "effective_samples": float(participation_ratio(profiles["sample"][:, None])[0]),
           "top_sender": celltypes[int(np.argmax(np.abs(profiles["sender"])))],
           "top_receiver": celltypes[int(np.argmax(np.abs(profiles["receiver"])))]}  # fmt: skip
    return row, profiles["sample"], lr


def _top_pair_genes(rows: list[dict]) -> list[str]:
    return sorted({g for r in rows for side in r["top_pair"].split("^") for g in side.split(COMPLEX_SEP)})


def audit_annotation(spec, adata, annotation, rise_fits, rng) -> pd.DataFrame:
    """Every missed column of one annotation's fits, classed, with its
    outcome association and the cccRISE rank that contains it."""
    suffix, column = ANNOTATIONS[annotation]
    fits, info, conditions = read_fits(VALIDATION_DIR / spec.benchmark_fits_filename(suffix))
    own = {f.method: f for f in fits if f.rank_mode == "own" and f.method != "cccRISE"}
    pairs = np.asarray(info["lr_pairs"]).astype(str)
    records = [_column_record(item, own[item.method], pairs, info["celltypes"])
               for item in missed_columns(spec, annotation).itertuples()]  # fmt: skip
    rows, samples, lr_columns = (list(x) for x in zip(*records, strict=True))
    fold, detected = marker_table(adata, column, _top_pair_genes(rows))
    fractions = _celltype_fractions(adata, spec, column).reindex(conditions)
    for row, sample in zip(rows, samples, strict=True):
        row["annotation"] = annotation
        row["class"] = classify(row, fold, detected, fractions, sample)
    table = pd.DataFrame(rows)
    stats = outcome_stats(adata, spec, np.column_stack(samples))
    for stat, outcome in stats.columns:
        table[f"{stat}: {outcome}"] = stats[(stat, outcome)].to_numpy()
    for tag, factors in rise_fits.items():
        table[f"contained_from_rank: {tag}"] = reach(
            np.column_stack(lr_columns), pairs, factors, spec.output_key, rng
        )
    return table


def run(spec: DatasetSpec, rise_factors: list[Path]) -> pd.DataFrame:
    adata = add_cond_idxs(spec.load(), spec.condition_column)
    rise_fits = {"canonical": spec.factors_path()} | {p.stem: p for p in rise_factors}
    rng = np.random.default_rng(0)
    table = pd.concat([audit_annotation(spec, adata, a, rise_fits, rng) for a in ANNOTATIONS],
                      ignore_index=True)  # fmt: skip
    out = VALIDATION_DIR / f"comparator_audit_{spec.output_key}.parquet"
    table.to_parquet(out, index=False)
    write_metadata(out, build_metadata(dataset_metadata(spec) | {
        "rise_factors": {tag: {"path": str(p), "sha256": sha256_file(p)} for tag, p in rise_fits.items()},
        "contained": CONTAINED, "one_sample": ONE_SAMPLE, "marker_fold": MARKER_FOLD,
        "marker_detected": MARKER_DETECTED, "abundance_rho": ABUNDANCE_RHO, "outcomes": list(OUTCOMES),
    }), header=HEADER)  # fmt: skip
    print(table.groupby(["annotation", "method", "class"]).size().to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_dataset_argument(parser)
    parser.add_argument("--rise-factors", type=Path, nargs="*", default=[],
                        help="Further RISE fits (factor files) whose CP rank sweeps to search.")  # fmt: skip
    args = parser.parse_args()
    run(resolve_dataset(args), args.rise_factors)


if __name__ == "__main__":
    main()
