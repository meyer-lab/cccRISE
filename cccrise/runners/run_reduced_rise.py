#!/usr/bin/env python
"""
Runner: CCC-RISE refit at a lower RISE rank, for the label-sensitivity
panels of Figure 5 (E-F, `figure5._coarse_fine`).

The comparators' coarse -> fine test asks what a label-based method keeps and
loses when its cell-type labels get finer. CCC-RISE has no labels; its
counterpart is the RISE rank, the number of eigen-states its cells are
described by. This runner refits the whole pipeline (`run_ccc_rise`'s
settings: uncompressed RISE, CP from the production SVD start on the
default device and precision) at `--rise-rank` and `--cp-rank` and writes
only the fit's LR pair loadings, which is all Figure 5E-F reads, so no
model file is kept:

    reduced_rise_lr_loadings_<dataset>.parquet
        one row per LR pair and component ('Method' cccRISE, 'RiseRank',
        'CpdRank', 'Component', 'LRPair', 'Loading'),

beside the comparator scores (`COMPARATOR_SCORES_DIR`), recorded in that
folder's `provenance.yaml` under its header (`run_comparator_scores.HEADER`).
The CPD rank is selected from a CPD rank sweep on the reduced fit
(`run_cpd_rank` on its factors), by the procedure used for the reported fits.

    uv run cccrise-reduced-rise --dataset alad --rise-rank 10 --cp-rank 4
"""

import argparse

import pandas as pd

from ..ccc_rise import CP_DTYPE, CP_FIT_OPTIONS, run_ccc_rise_workflow
from ..import_data import import_ligand_receptor_pairs
from ._cli import (
    CP_N_ITER_MAX,
    CP_TOL,
    RISE_N_ITER_MAX,
    RISE_TOL,
    add_dataset_argument,
    cp_device_metadata,
    dataset_metadata,
    parse_command,
    resolve_dataset,
)
from .datasets import COMPARATOR_SCORES_DIR, DatasetSpec, scores_filename
from .provenance import build_metadata, git_sha, read_metadata, record_output
from .run_comparator_scores import HEADER

TABLE = "reduced_rise_lr_loadings"


def lr_loading_table(fit: dict, rise_rank: int, cp_rank: int) -> pd.DataFrame:
    """The CP fit's LR pair mode, long: one row per LR pair and component."""
    pairs = fit["filtered_lr_pairs"]["interaction_symbol"].astype(str).to_numpy()
    lr = fit["cp_factors"][3]
    return (
        pd.DataFrame(lr, index=pairs, columns=range(1, lr.shape[1] + 1))
        .rename_axis(index="LRPair", columns="Component")
        .stack()
        .rename("Loading")
        .reset_index()
        .assign(Method="cccRISE", RiseRank=rise_rank, CpdRank=cp_rank)
    )[["Method", "RiseRank", "CpdRank", "Component", "LRPair", "Loading"]]


def run_reduced_rise(
    spec: DatasetSpec,
    rise_rank: int,
    cp_rank: int,
    command: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit CCC-RISE at `rise_rank` / `cp_rank` and write its LR loadings."""
    git_sha()
    print(f"=== Reduced RISE: {spec.label}, RISE {rise_rank} / CPD {cp_rank} ===")
    reported = read_metadata(spec.factors_path())["rank_selection"]
    fit = run_ccc_rise_workflow(
        adata=spec.load(),
        rise_rank=rise_rank,
        lr_pairs=import_ligand_receptor_pairs(),
        cp_rank=cp_rank,
        condition_column=spec.condition_column,
        random_state=random_state,
        backend=spec.backend,
        compress=False,
        cp_dtype=CP_DTYPE,
        rise_n_iter_max=RISE_N_ITER_MAX,
        cp_n_iter_max=CP_N_ITER_MAX,
        rise_tol=RISE_TOL,
        cp_tol=CP_TOL,
    )
    print(f"   RISE R2X {fit['r2x_rise']:.4f}, CP R2X {fit['r2x_cp']:.4f}")
    table = lr_loading_table(fit, rise_rank, cp_rank)
    path = COMPARATOR_SCORES_DIR / scores_filename(TABLE, spec.output_key)
    table.to_parquet(path, index=False)
    metadata = build_metadata(
        dataset_metadata(spec)
        | {
            "rise_rank": rise_rank,
            "cp_rank": cp_rank,
            "reported_fit": {
                "factors_file": spec.factors_filename,
                "rise_rank": reported["rise"]["rank"],
                "cp_rank": reported["cpd"]["rank"],
            },
            "random_state": random_state,
            "rise_compressed": False,
            "rise_backend": spec.backend or "auto",
            "rise_n_iter_max": RISE_N_ITER_MAX,
            "rise_tol": RISE_TOL,
            "cp_n_iter_max": CP_N_ITER_MAX,
            "cp_tol": CP_TOL,
            "cp_fit_options": CP_FIT_OPTIONS,
            **cp_device_metadata(None, CP_DTYPE),
            "r2x_rise": float(fit["r2x_rise"]),
            "r2x_cp": float(fit["r2x_cp"]),
        }
    )
    record_output(path, metadata, command, header=HEADER)
    print(f"   wrote {path.name}")
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_dataset_argument(parser)
    parser.add_argument("--rise-rank", type=int, required=True)
    parser.add_argument("--cp-rank", type=int, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args, command = parse_command(build_parser(), "cccrise-reduced-rise", argv)
    run_reduced_rise(
        resolve_dataset(args),
        args.rise_rank,
        args.cp_rank,
        command,
        args.random_state,
    )


if __name__ == "__main__":
    main()
