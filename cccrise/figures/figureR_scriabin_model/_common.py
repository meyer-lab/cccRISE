"""Shared inputs of the reviewer figure on Scriabin's seed stability: the
tables of ``uv run cccrise-scriabin-stability`` (`run_scriabin_stability`),
and how preprocessings and datasets are named, colored and drawn."""

import numpy as np
import pandas as pd

from ...runners.datasets import SCRIABIN_DIR
from .. import common, style

HINT = (
    "    uv run cccrise-scriabin-stability --analysis seed_agreement --dataset <alad|ibd|panc8>\n"
    "    uv run cccrise-scriabin-stability --analysis remedy --dataset <alad|panc8>"
)
_BLUE, _ORANGE, _GREEN, _VERMILLION, _, _TAN, *_ = style.CATEGORICAL
#: Each preprocessing's color, and its label: the steps that differ, then
#: (after the arrow, in A's legend) its LR resource.
PREPROCESSING_COLORS = {"cccrise": _BLUE, "sct_alra": _GREEN, "sct": _TAN}
PREPROCESSING_LABELS = {
    "cccrise": r"Depth and gene-sum scaling, $\mathregular{log_{10}(1000 \cdot \tilde{x} + 1)}$",
    "sct_alra": "SCTransform + ALRA",
    "sct": "SCTransform",
}
LR_RESOURCES = {"cccrise": "CellChatDB", "sct_alra": "OmniPath", "sct": "OmniPath"}
#: (dataset, grouping) -> row label, and its marker and line style in B
DATASET_LABELS = {
    ("alad", "sample"): "ALAD",
    ("ibd", "sample"): "IBD",
    ("panc8", "sample"): "panc8, per donor",
    ("panc8", "celltype"): "panc8, per cell type",
}
DATASET_STYLES = {
    "ALAD": ("o", "-"),
    "IBD": ("s", (0, (1, 1.2))),
    "panc8, per donor": ("^", (0, (4, 1.5))),
    "panc8, per cell type": ("D", (0, (4, 1.2, 1, 1.2))),
}
#: Each Scriabin run of A and B, top to bottom: (dataset, grouping,
#: preprocessing)
SCRIABIN_RUNS = (
    ("alad", "sample", "cccrise"),
    ("alad", "sample", "sct_alra"),
    ("alad", "sample", "sct"),
    ("ibd", "sample", "cccrise"),
    ("panc8", "sample", "cccrise"),
    ("panc8", "sample", "sct_alra"),
    ("panc8", "celltype", "cccrise"),
    ("panc8", "celltype", "sct_alra"),
)
REMEDY_DATASETS = {"alad": "ALAD", "panc8": "panc8, per cell type"}
REMEDY_COLORS = {"alad": _ORANGE, "panc8": _VERMILLION}
INK = style.COLORS["ink"]


def load(analysis: str, dataset: str) -> pd.DataFrame:
    return common.load_runner_output(
        f"{analysis}_{dataset}.parquet", SCRIABIN_DIR, HINT
    )


def seed_agreement() -> pd.DataFrame:
    """Every dataset's seed-agreement rows, with its ``Dataset``."""
    return pd.concat(
        [load("seed_agreement", d).assign(Dataset=d) for d in ("alad", "ibd", "panc8")],
        ignore_index=True,
    )


def run_rows(
    table: pd.DataFrame, dataset: str, grouping: str, preprocessing: str, method: str
) -> pd.DataFrame:
    return table[
        (table.Dataset == dataset)
        & (table.Grouping == grouping)
        & (table.Preprocessing == preprocessing)
        & (table.Method == method)
    ]


def brace(
    ax,
    a: float,
    b: float,
    at: float,
    depth: float,
    vertical: bool = False,
    transform=None,
) -> float:
    """A curly brace spanning `a`..`b` at offset `at`, its tip `depth`
    further out (negative: down or left); returns the tip's offset."""
    t = np.linspace(0, 0.5, 80)
    k = 40.0  # sharpness of the brace's bends
    half = 1 / (1 + np.exp(-k * t)) + 1 / (1 + np.exp(-k * (t - 0.5))) - 0.5
    g = np.concatenate([half, half[-2::-1]])
    pos = a + (b - a) * np.concatenate([t, 1 - t[-2::-1]])
    off = at + depth * g
    xy = (off, pos) if vertical else (pos, off)
    ax.plot(
        *xy,
        transform=transform,
        color=INK,
        lw=style.LINE_WIDTH_THIN * 1.2,
        clip_on=False,
    )
    return at + depth


def extent(text, ax) -> np.ndarray:
    """`text`'s drawn extent, [[x0, y0], [x1, y1]] in `ax`'s axes coordinates."""
    renderer = ax.figure.canvas.get_renderer()
    return ax.transAxes.inverted().transform(text.get_window_extent(renderer))
