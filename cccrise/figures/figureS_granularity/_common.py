"""Shared loading and drawing for the granularity figure: the ladder tables of
`run_granularity_ladder` and `run_granularity_summary`."""

import pandas as pd

from ...runners.datasets import VALIDATION_DIR
from .. import common, style

DATASET = "alad"
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
SHORT = {"Pseudobulk CPD": "PB CPD", "Tensor-cell2cell": "TC2C", "STACCato": "STACCato"}


LADDER_HINT = (
    f"    uv run python -m cccrise.runners.run_granularity_ladder --dataset {DATASET}"
)


def load_ladder(name: str) -> pd.DataFrame:
    """One `run_granularity_ladder` table: representability, separation,
    dominance or strip."""
    return common.load_runner_output(
        f"granularity_{name}_{DATASET}.parquet", VALIDATION_DIR, hint=LADDER_HINT
    )


FITS_HINT = (
    f"    uv run python -m cccrise.runners.run_granularity_summary --dataset {DATASET}"
)
TICKS = [8, 13, 23, 38]


def load_fits() -> pd.DataFrame:
    """`run_granularity_summary`'s one row per comparator fit and level."""
    return common.load_runner_output(
        f"granularity_fits_{DATASET}.parquet", VALIDATION_DIR, hint=FITS_HINT
    )


def ladder_series(table, method, rank_mode, kind, column):
    """(n_types, values) from the coarse annotation through one ladder kind
    ("expr" or "rand"), for one method and rank mode."""
    sub = table[(table["method"] == method) & (table["rank_mode"] == rank_mode)]
    rows = sub[(sub["level"] == "coarse") | (sub["kind"] == kind)].sort_values(
        "n_types"
    )
    return rows["n_types"].to_numpy(), rows[column].to_numpy(float)


def draw_ladder(ax, table, method, column, color, rank_mode="shared"):
    """One method's ladder at `rank_mode`: expression subtypes solid, random
    splits dashed, the fine annotation a star; own-rank fits as open marks."""
    x, y = ladder_series(table, method, rank_mode, "expr", column)
    ax.plot(x, y, color=color, marker="o", ms=style.MARKER_SIZE, lw=style.LINE_WIDTH)
    x, y = ladder_series(table, method, rank_mode, "rand", column)
    ax.plot(x, y, color=color, ls="--", lw=style.LINE_WIDTH_THIN)
    sub = table[(table["method"] == method) & (table["level"] == "fine")]
    for mode, marker in ((rank_mode, "*"), ("own", "D")):
        row = sub[sub["rank_mode"] == mode]
        if len(row):
            ax.scatter(
                row["n_types"],
                row[column],
                marker=marker,
                s=40 if marker == "*" else 10,
                color=color,
                zorder=4,
                facecolor=color if marker == "*" else "none",
            )
    own = table[
        (table["method"] == method)
        & (table["rank_mode"] == "own")
        & (table["kind"] != "annotation")
    ]
    ax.scatter(
        own["n_types"],
        own[column],
        marker="o",
        s=12,
        facecolor="none",
        edgecolor=color,
        lw=0.6,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xticks(TICKS, [str(t) for t in TICKS])
    ax.minorticks_off()
    ax.grid(False)
