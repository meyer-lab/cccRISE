import argparse
from pathlib import Path

from .datasets import DATASET_KEYS, DatasetSpec
from .provenance import sha256_file


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
        if ":" in chunk:
            ranks.update(range(*(int(p) for p in chunk.split(":"))))
        else:
            ranks.add(int(chunk))
    return sorted(ranks)


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_KEYS,
        help="Dataset label from the runner registry.",
    )


def dataset_metadata(spec: DatasetSpec) -> dict:
    """The dataset-identity head of a sidecar."""
    meta: dict = {
        "dataset": spec.key,
        "dataset_label": spec.label,
        "source_file": str(spec.source_file),
        "source_sha256": sha256_file(spec.source_file),
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
    """Dimensions of the data a run actually saw."""
    dims: dict = {
        "conditions": int(adata.obs[spec.condition_column].nunique()),
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
