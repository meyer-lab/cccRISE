"""Small argument-parsing helpers shared by the runner CLIs."""

import argparse
from pathlib import Path

from .datasets import DATASET_KEYS, DatasetSpec, get_dataset
from .provenance import source_info


def parse_ranks(spec: str) -> list[int]:
    """Parse a rank specification into a sorted list of unique ranks.

    Accepts comma-separated values and ``start:stop[:step]`` slices (``stop``
    exclusive, matching `range`), mixed freely::

        "5,10,15"    -> [5, 10, 15]
        "1:26"       -> [1, 2, ..., 25]
        "5:51:5"     -> [5, 10, ..., 50]
        "1:6,10,20"  -> [1, 2, 3, 4, 5, 10, 20]
    """
    ranks: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(p) for p in chunk.split(":")]
            if len(parts) == 2:
                ranks.update(range(parts[0], parts[1]))
            elif len(parts) == 3:
                ranks.update(range(parts[0], parts[1], parts[2]))
            else:
                raise argparse.ArgumentTypeError(
                    f"Bad rank slice {chunk!r}; expected start:stop[:step]."
                )
        else:
            ranks.add(int(chunk))
    if not ranks:
        raise argparse.ArgumentTypeError(f"No ranks parsed from {spec!r}.")
    if min(ranks) < 1:
        raise argparse.ArgumentTypeError(f"Ranks must be >= 1; got {sorted(ranks)}.")
    return sorted(ranks)


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--dataset`` / ``--gene-threshold`` options."""
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_KEYS,
        help="Dataset label from the runner registry.",
    )
    parser.add_argument(
        "--gene-threshold",
        type=float,
        default=None,
        help="Mean-expression gene filter; defaults to the dataset's registry value.",
    )
    parser.add_argument(
        "--condition-column",
        type=str,
        default=None,
        help="Override the dataset's condition/sample column in adata.obs.",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default=None,
        help="Override the raw .h5ad the loader reads. Recorded, with its "
        "sha256, in the provenance sidecar.",
    )
    parser.add_argument(
        "--skip-source-checksum",
        action="store_true",
        help="Skip hashing the source .h5ad. Faster, but drops the field that "
        "detects source-data drift (issue #164).",
    )


def resolve_dataset(args: argparse.Namespace) -> DatasetSpec:
    """Apply ``--condition-column`` / ``--gene-threshold`` overrides to the spec."""
    from dataclasses import replace

    spec = get_dataset(args.dataset)
    overrides: dict = {}
    if getattr(args, "condition_column", None):
        overrides["condition_column"] = args.condition_column
    if getattr(args, "gene_threshold", None) is not None:
        overrides["gene_threshold"] = args.gene_threshold
    if getattr(args, "source_file", None):
        overrides["source_file"] = Path(args.source_file)
    return replace(spec, **overrides) if overrides else spec


def dataset_metadata(spec: DatasetSpec, checksum_source: bool = True) -> dict:
    """The dataset-identity head of a sidecar, in the key order issue #164 documents.

    Together these answer "what data, read how?" -- the question the stored
    ``.nc`` files could not answer, which is how the ALAD artifact drifted from
    445 to 513 LR pairs without anyone noticing.
    """
    meta: dict = {"dataset": spec.key, "dataset_label": spec.label}
    meta |= source_info(spec.source_file, checksum=checksum_source)
    meta |= {
        "loader": spec.loader_path,
        "loader_args": spec.loader_args,
        "condition_key": spec.condition_column,
    }
    if spec.celltype_column:
        meta["celltype_key"] = spec.celltype_column
    if spec.group_column:
        meta["group_key"] = spec.group_column
    if spec.notes:
        meta["dataset_notes"] = spec.notes
    return meta


def resulting_dims(spec: DatasetSpec, adata, n_lr_pairs: int | None = None) -> dict:
    """``resulting_dims`` per issue #164: the dimensions that expose source drift.

    Conditions, genes and LR pairs are exactly the axes on which the stored
    ALAD factors failed to reproduce, so they are recorded explicitly rather
    than left to be inferred from the tensor shape.
    """
    dims: dict = {
        "conditions": (
            int(adata.obs[spec.condition_column].nunique())
            if spec.condition_column in adata.obs
            else None
        ),
        "genes": int(adata.n_vars),
        "cells": int(adata.n_obs),
    }
    if n_lr_pairs is not None:
        dims["lr_pairs"] = int(n_lr_pairs)
    return dims


def resolve_output(explicit: str | None, default_dir: Path, default_name: str) -> Path:
    """Return the output path, defaulting to ``default_dir / default_name``."""
    path = Path(explicit) if explicit else default_dir / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
