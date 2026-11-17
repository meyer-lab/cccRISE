"""Shared inputs of the reviewer figure on the Scriabin paper's technical-noise
analysis (Wilk et al. 2024, Fig. 2d-e): the tables of ``uv run
cccrise-scriabin-reproduction`` (`run_scriabin_reproduction`)."""

import numpy as np
import pandas as pd

from ...runners.datasets import SCRIABIN_REPRODUCTION_DIR
from .. import common, style

HINT = "    uv run cccrise-scriabin-reproduction --analysis <downsampling|downsampling_programs>"
INK = style.COLORS["ink"]
#: The paper's marks, drawn as references beside ours
PAPER = style.COLORS["reference"]


def load(analysis: str) -> pd.DataFrame:
    return common.load_runner_output(
        f"{analysis}_panc8.parquet", SCRIABIN_REPRODUCTION_DIR, HINT
    )


def jitter(values, width: float, seed: int) -> np.ndarray:
    """`values` plus uniform jitter of ±`width`, from its own generator so
    every build draws the same."""
    values = np.asarray(values, float)
    return values + np.random.default_rng(seed).uniform(-width, width, len(values))


def boxplot(ax, values: list, width: float) -> None:
    """Thin-lined boxes without fliers (the points are drawn over them), one
    per entry of `values` at 0, 1, ..."""
    ax.boxplot(
        values,
        positions=range(len(values)),
        widths=width,
        showfliers=False,
        medianprops={"color": INK, "lw": style.LINE_WIDTH},
        boxprops={"lw": style.LINE_WIDTH_THIN},
        whiskerprops={"lw": style.LINE_WIDTH_THIN},
        capprops={"lw": style.LINE_WIDTH_THIN},
    )
