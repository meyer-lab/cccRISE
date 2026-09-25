import argparse
from pathlib import Path

from .datasets import DATASET_KEYS, DatasetSpec
from .provenance import sha256_file

# (control label, other label) for `figures.commonFuncs.plotGeneral.binarize_group`,
# per dataset. Only datasets with a confirmed binary encoding of `group_column`
# are listed; an unlisted dataset requires --control-label/--other-label
# explicitly.
DEFAULT_GROUP_LABELS: dict[str, tuple[str, str]] = {
    "alad": ("control", "alad"),
    "ibd": ("healthy", "inflamed"),
}

# Fit controls shared by every runner's RISE (PARAFAC2) and CP decompositions.
RISE_TOL = 1e-8
CP_TOL = 1e-12
RISE_N_ITER_MAX = 300
CP_N_ITER_MAX = 4000


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


def add_group_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the binary-outcome arguments every group-comparison runner takes."""
    parser.add_argument(
        "--group-column",
        type=str,
        default=None,
        help="Column in adata.obs carrying the binary group label "
        "(default: the dataset registry's group_column).",
    )
    parser.add_argument(
        "--control-label",
        type=str,
        default=None,
        help="Value of --group-column mapped to the reference class "
        "(default: dataset-specific, see DEFAULT_GROUP_LABELS).",
    )
    parser.add_argument(
        "--other-label",
        type=str,
        default=None,
        help="Label applied to every other value of --group-column "
        "(default: dataset-specific, see DEFAULT_GROUP_LABELS).",
    )


def resolve_group_labels(
    spec: DatasetSpec,
    group_column: str | None,
    control_label: str | None,
    other_label: str | None,
) -> tuple[str, str | None, str | None]:
    """Fill a runner's group encoding in from the dataset registry.

    When both labels are omitted and the dataset has a registered default
    binary encoding in `DEFAULT_GROUP_LABELS`, those defaults are used.
    When omitted and no default exists, returns ``(group_column, None, None)``
    to test every level.

    Returns
    -------
    tuple[str, str | None, str | None]
        The group column, the control label, and the other label.
    """
    group_column = group_column or spec.group_column
    if group_column is None:
        raise ValueError(
            f"Dataset {spec.key!r} has no group_column in the registry; "
            "pass --group-column explicitly."
        )
    if control_label is None and other_label is None:
        defaults = DEFAULT_GROUP_LABELS.get(spec.key)
        if defaults is not None:
            return group_column, defaults[0], defaults[1]
        return group_column, None, None
    if control_label is None or other_label is None:
        raise ValueError(
            f"Pass both --control-label and --other-label to collapse to two named "
            f"levels, or neither to test every level of {group_column!r}; got "
            f"control_label={control_label!r}, other_label={other_label!r}."
        )
    return group_column, control_label, other_label


def dataset_metadata(spec: DatasetSpec) -> dict:
    """The dataset-identity head of a sidecar."""
    meta: dict = {
        "dataset": spec.key,
        "dataset_label": spec.label,
        "source_file": str(spec.source_file),
        "source_sha256": sha256_file(spec.source_file),
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
