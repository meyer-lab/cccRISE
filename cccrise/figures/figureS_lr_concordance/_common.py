"""Shared loading for the LR-concordance figure (`run_lr_concordance`):
own-rank comparator fits on the coarse and fine annotations."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common

DATASET = "alad"
HINT = f"    uv run python -m cccrise.runners.run_lr_concordance --dataset {DATASET}"
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
SHORT = {"Pseudobulk CPD": "PB", "Tensor-cell2cell": "TC2C", "STACCato": "STAC"}
ANNOTATIONS = ("coarse", "fine")
CLASSES = (
    "recovered",
    "same LR, different who/where",
    "not recovered, one sample",
    "not recovered, broad",
)


def load(name: str, own_only: bool = True) -> pd.DataFrame:
    """One `lr_concordance` table, own-rank comparators (and cccRISE /
    the abundance reference) unless `own_only` is False."""
    table = common.load_runner_output(
        f"lr_concordance_{name}_{DATASET}.parquet", VALIDATION_DIR, hint=HINT
    )
    if own_only and "rank_mode" in table:
        table = table[table["rank_mode"] != "shared"]
    return table.reset_index(drop=True)


def column_order(table: pd.DataFrame) -> list[tuple[str, str, int]]:
    """(annotation, method, component) of every comparator column, grouped
    by annotation, then method."""
    keys = table[["annotation", "method", "comparator_component"]].drop_duplicates()
    keys = keys[keys["method"].isin(METHODS)]
    return sorted(
        keys.itertuples(index=False, name=None),
        key=lambda k: (ANNOTATIONS.index(k[0]), METHODS.index(k[1]), k[2]),
    )


def top_pair(columns: pd.DataFrame) -> dict[tuple[str, str, int], str]:
    """Each LR column's heaviest pair, keyed like `column_order`."""
    first = columns["top_pairs"].str.split(";").str[0].str.rsplit(" ", n=1).str[0]
    return dict(
        zip(
            zip(
                columns["annotation"],
                columns["method"],
                columns["component"],
                strict=True,
            ),
            first,
            strict=True,
        )
    )


def column_label(key: tuple[str, str, int], pairs: dict) -> str:
    annotation, method, component = key
    return f"{SHORT[method]}{annotation[0]}{component} {pairs.get(key, '')}"
