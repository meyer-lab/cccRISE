#!/usr/bin/env python
"""
Runner: the NicheNet tests on cccRISE and every comparator LR column of a cohort.

Each column gets a high and a low group of cells and the target scores of every
ligand on their differential expression (`nichenet.component_ligand_scores`),
tested against rankings with the two groups' labels shuffled among their cells
(Figure 5G).

- **cccRISE:** as Figure 2, the top receiver cells of each component's receiver
  cell types against the bottom cells of the dataset.
- **Comparators:** a column has one receiver value per label, so no cells can be
  taken within a label. High is every cell of the labels at or above
  `nichenet.rel_cutoff` of the largest, low the lowest labels up to `TOP_PERC`
  percent of cells. A column whose labels all clear the cutoff has no low group
  and names no receiver; it takes the top labels up to `TOP_PERC` percent
  instead (``Rule`` "fallback").

Writes ``nichenet_<dataset>.parquet`` beside the comparator scores, one row per
column, adjusted within each fit.

    uv run cccrise-comparator-nichenet --dataset alad
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
import xarray as xr

from ..comparators import scores
from ..factors import load_factors_to_anndata
from ..validation import nichenet
from ._cli import (
    add_dataset_argument,
    dataset_metadata,
    parse_command,
    resolve_dataset,
)
from .datasets import (
    COMPARATOR_FITS_DIR,
    COMPARATOR_SCORES_DIR,
    DatasetSpec,
    comparator_filename,
    scores_filename,
)
from .provenance import build_metadata, git_sha, record_output
from .run_comparator_scores import HEADER

METHODS = {"pb": "PB", "tc2c": "TC2C", "stac": "STAC"}
TOP_PERC = 10.0
REL_CUTOFF = 0.4
N_SHUFFLE = 1000
#: Filled before forking, so workers share the cells.
_SHARED: dict = {}


def receiver_profile(fit: xr.Dataset, column: int) -> pd.Series:
    """A comparator column's receiver values by label, its largest positive."""
    if "core" in fit:
        values = scores.tucker_column_profiles(fit, column)["receiver"]
    else:
        values = fit["receiver_factor"].to_numpy()[:, column - 1]
    names = fit["receiver"].to_numpy().astype(str)
    return scores.oriented(pd.Series(values, index=names).to_frame()).iloc[:, 0]


def _fill(order, skip: list, labels: np.ndarray) -> tuple[list, np.ndarray]:
    """Labels of `order`, past `skip`, until they reach `TOP_PERC` percent of cells."""
    taken, mask = [], np.zeros(len(labels), dtype=bool)
    for label in order:
        if label in skip:
            continue
        if mask.mean() * 100 >= TOP_PERC:
            break
        taken.append(label)
        mask |= labels == label
    return taken, mask


def label_groups(profile: pd.Series, labels: np.ndarray) -> dict:
    """A comparator column's high and low cells, and the rule that chose them."""
    high = list(profile.index[profile >= REL_CUTOFF * profile.max()])
    rule = "cutoff"
    _, low = _fill(profile.sort_values().index, high, labels)
    if not low.any():
        rule = "fallback"
        high, _ = _fill(profile.sort_values(ascending=False).index, [], labels)
        _, low = _fill(profile.sort_values().index, high, labels)
    return {"high": np.isin(labels, high), "low": low, "Rule": rule}


def _score_column(job: tuple) -> tuple:
    """A column's observed ranking and its shuffled target scores."""
    key, column, high, low = job
    kept = high | low
    pos = high[kept]
    masks = np.vstack(
        [pos]
        + [np.random.default_rng(seed).permutation(pos) for seed in range(N_SHUFFLE)]
    )
    genes = _SHARED["genes"]
    z, frac_high, frac_low = nichenet.rank_sum_statistics(_SHARED["X"][kept], masks)
    rankings = (
        nichenet.statistic_ranking(z[i], frac_high[i], frac_low[i], genes)
        for i in range(1, len(masks))
    )
    null = pd.concat(
        nichenet.ligand_activities(ranking, _SHARED["prior"])
        .rename("auroc")
        .rename_axis("ligand")
        .reset_index()
        .assign(component=column, replicate=seed)
        for seed, ranking in enumerate(rankings)
    )
    observed = nichenet.statistic_ranking(z[0], frac_high[0], frac_low[0], genes)
    return key, column, observed, int(pos.sum()), int((~pos).sum()), null


def _fit_tests(key, columns: dict, D, lr_pairs, prior) -> pd.DataFrame:
    """The cell-shuffle test of one fit's columns, adjusted within it."""
    rankings = {c: v[:3] for c, v in columns.items()}
    ligand_scores = nichenet.component_ligand_scores(
        D, lr_pairs, prior, rankings, REL_CUTOFF
    )
    cell_null = pd.concat([v[3] for v in columns.values()], ignore_index=True)
    cells = nichenet.weighted_score_tests(ligand_scores, cell_null)[0]
    return cells.rename(
        columns={
            "component": "Component",
            "null_mean": "null_mean_cell",
            "p": "p_cell",
            "q": "q_cell",
        }
    ).assign(Method=key[0], Labelling=key[1])


def _lr_pairs(names) -> pd.DataFrame:
    ligand, receptor = zip(*(str(s).split("^", 1) for s in names), strict=True)
    return pd.DataFrame({"ligand": ligand, "receptor": receptor})


def _comparator_jobs(spec: DatasetSpec, adata) -> tuple[list, list, dict]:
    jobs, groups, fits = [], [], {}
    for key, method in METHODS.items():
        for lab, column in spec.labellings.items():
            path = COMPARATOR_FITS_DIR / comparator_filename(
                key, "fit", spec.output_key, lab
            )
            fit = xr.open_dataset(path).load()
            fits[(method, lab)] = fit
            labels = adata.obs[column].astype(str).to_numpy()
            for c in range(1, fit["lr_factor"].shape[1] + 1):
                chosen = label_groups(receiver_profile(fit, c), labels)
                jobs.append(((method, lab), c, chosen["high"], chosen["low"]))
                groups.append(
                    {
                        "Method": method,
                        "Labelling": lab,
                        "Component": c,
                        "Rule": chosen["Rule"],
                    }
                )
    return jobs, groups, fits


def _cccrise_jobs(spec: DatasetSpec, adata) -> tuple[list, list]:
    jobs, groups = [], []
    for c in range(1, adata.uns["D"].shape[1] + 1):
        high, low = nichenet.receiver_masks(adata, c, TOP_PERC, spec.celltype_column)
        jobs.append((("cccRISE", "coarse"), c, high, low))
        groups.append(
            {
                "Method": "cccRISE",
                "Labelling": "coarse",
                "Component": c,
                "Rule": "cells",
            }
        )
    return jobs, groups


def run_comparator_nichenet(
    spec: DatasetSpec, command: str, n_jobs: int = 12
) -> pd.DataFrame:
    git_sha()
    adata = load_factors_to_anndata(spec.factors_path(), adata=spec.load())
    prior = nichenet.load_ligand_target_matrix()
    _SHARED.update(X=adata.X, genes=adata.var_names.to_numpy(), prior=prior)
    jobs, groups, fits = _comparator_jobs(spec, adata)
    cccrise_jobs, cccrise_groups = _cccrise_jobs(spec, adata)
    results: dict = {}
    with ProcessPoolExecutor(n_jobs, mp_context=get_context("fork")) as pool:
        for key, column, ranking, n_high, n_low, null in pool.map(
            _score_column, [*cccrise_jobs, *jobs]
        ):
            results.setdefault(key, {})[column] = (ranking, n_high, n_low, null)
            print(f"   {key} {column}", flush=True)
    frames = [
        _fit_tests(
            ("cccRISE", "coarse"),
            results[("cccRISE", "coarse")],
            adata.uns["D"],
            _lr_pairs(adata.uns["lr_pairs"]),
            prior,
        )
    ]
    for key, fit in fits.items():
        frames.append(
            _fit_tests(
                key,
                results[key],
                np.nan_to_num(fit["lr_factor"].to_numpy()),
                _lr_pairs(fit["lr"].to_numpy()),
                prior,
            )
        )
    sizes = pd.DataFrame(
        [
            {
                "Method": key[0],
                "Labelling": key[1],
                "Component": column,
                "High cells": v[1],
                "Low cells": v[2],
            }
            for key, columns in results.items()
            for column, v in columns.items()
        ]
    )
    keys = ["Method", "Labelling", "Component"]
    table = (
        pd.concat(frames, ignore_index=True)
        .merge(pd.DataFrame(cccrise_groups + groups), on=keys)
        .merge(sizes, on=keys)
    )
    path = COMPARATOR_SCORES_DIR / scores_filename("nichenet", spec.output_key)
    table.to_parquet(path, index=False)
    metadata = build_metadata(
        dataset_metadata(spec)
        | {
            "top_perc": TOP_PERC,
            "rel_cutoff": REL_CUTOFF,
            "n_shuffle": N_SHUFFLE,
        }
    )
    record_output(path, metadata, command, header=HEADER)
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_dataset_argument(parser)
    parser.add_argument("--n-jobs", type=int, default=min(12, os.cpu_count() or 1))
    return parser


def main(argv: list[str] | None = None) -> None:
    args, command = parse_command(build_parser(), "cccrise-comparator-nichenet", argv)
    run_comparator_nichenet(resolve_dataset(args), command, args.n_jobs)


if __name__ == "__main__":
    main()
