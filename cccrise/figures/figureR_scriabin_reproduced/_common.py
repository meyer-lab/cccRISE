"""Shared inputs of the reviewer figure reproducing the Scriabin paper's
robustness analyses (Wilk et al. 2024): the tables of ``uv run
cccrise-scriabin-reproduction`` (`run_scriabin_reproduction`), and the
paper's own values each panel is compared with."""

import numpy as np
import pandas as pd
from scipy import stats

from ...runners.datasets import SCRIABIN_REPRODUCTION_DIR
from .. import common, style

HINT = "    uv run cccrise-scriabin-reproduction --analysis <downsampling|soft_power>"
INK = style.COLORS["ink"]
#: The paper's marks, drawn as references beside ours
PAPER = style.COLORS["reference"]
#: R prints p-values below machine precision this way (ggpubr's stat_cor)
P_FLOOR = 2.2e-16


def load(analysis: str) -> pd.DataFrame:
    return common.load_runner_output(
        f"{analysis}_panc8.parquet", SCRIABIN_REPRODUCTION_DIR, HINT
    )


def jitter(values, width: float, seed: int) -> np.ndarray:
    """`values` plus uniform jitter of ±`width`, from its own generator so
    every build draws the same."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    return values + rng.uniform(-width, width, len(values))


def pearson_label(x, y) -> str:
    """'R = r, p = p' as the paper's panels print it (Pearson)."""
    r, p = stats.pearsonr(np.asarray(x, float), np.asarray(y, float))
    p_text = f"p < {P_FLOOR:.1e}" if p < P_FLOOR else f"p = {p:.1e}"
    return f"R = {r:.2f}, {p_text}"


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


#: Extended Data Fig. 8a: the mean Jaccard between ligand-activity parameter
#: settings that the paper calls "generally stable results"
STABLE_BAND = (0.47, 0.71)


def stable_band(ax, label_x: float = 0.98) -> None:
    """The paper's own "generally stable" Jaccard range, shaded, labeled at
    `label_x` (axes fraction; right-aligned past the middle)."""
    low, high = STABLE_BAND
    ax.axhspan(low, high, color=style.COLORS["muted"], alpha=0.35, lw=0, zorder=0)
    ax.text(
        label_x,
        high - 0.015,
        f"Paper's “generally stable”\n(ED Fig. 8a, {low}–{high})",
        transform=ax.get_yaxis_transform(),
        ha="right" if label_x > 0.5 else "left",
        va="top",
        color=PAPER,
        fontsize=style.FONT_SIZE_DENSE,
    )
