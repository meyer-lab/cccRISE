"""Shared loaders and drawing helpers for the carrier exploration
(`run_hormone_cells`; `HORMONE_CELLS_PREREGISTRATION.md` and its ALAD
addendum). Every panel reads the runner's tables and per-cell caches; none
computes a statistic beyond a display transform (a group mean for the label
oracle). The dataset drawn is `DATASET`, set from the environment variable
``CELLTYPE_EXPLORATION_DATASET`` (default ``ibd``)."""

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from ...runners.datasets import VALIDATION_DIR
from ...runners.run_hormone_cells import CACHE_DIR, cells_path, factors_tag
from ...validation import hormone_cells as H
from .. import style

QUANTITY_LABELS = {
    "loading": "Loading",
    "contribution": "Contribution",
    "expressed": "Expressed part",
}
METHOD_SHORT = {"Pseudobulk CPD": "PB", "Tensor-cell2cell": "TC2C"}
DATASET = os.environ.get("CELLTYPE_EXPLORATION_DATASET", "ibd")
CFG = H.CONFIGS[DATASET]
#: pt^2 per cell: small for IBD's 634k cells, larger for ALAD's 51k
POINT = 0.15 if DATASET == "ibd" else 0.8
LEVEL_NAMES = ["0 none", "1 present", "2 material", "3 substantial"]


def table(name: str) -> pd.DataFrame:
    path = VALIDATION_DIR / f"hormone_cells_{name}_{DATASET}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing; run\n"
            f"    uv run python -m cccrise.runners.run_hormone_cells analyze --dataset {DATASET}"
        )
    return pd.read_parquet(path)


@functools.cache
def tag() -> str:
    return factors_tag(DATASET)[1]


@functools.cache
def cells() -> dict:
    keep = ("sample", "lineage", "celltype", "group", "inflammation", "projections")
    with np.load(cells_path(tag(), DATASET), allow_pickle=True) as z:
        out = {k: z[k] for k in keep if k in z.files}
    out = {k: v.astype(str) if v.dtype == object else v for k, v in out.items()}
    if "group" not in out:  # the IBD table predates the dataset option
        out["group"] = out.pop("inflammation")
    return out


@functools.cache
def values() -> dict:
    with np.load(CACHE_DIR / f"{DATASET}_values_{tag()}.npz") as z:
        return {k: z[k] for k in z.files}


@functools.cache
def pacmap(seed: int = 42) -> np.ndarray:
    with np.load(CACHE_DIR / f"{DATASET}_pacmap_{tag()}_seed{seed}.npz") as z:
        return z["embedding"]


def lineage_order() -> list[str]:
    return [t for t in style.CELL_TYPES[DATASET] if t in set(cells()["lineage"])]


def fine_order() -> list[str]:
    c = cells()
    parent = pd.Series(c["lineage"], index=c["celltype"]).groupby(level=0).first()
    rank = {lin: i for i, lin in enumerate(lineage_order())}
    return sorted(parent.index, key=lambda t: (rank[parent[t]], t))


def lineage_colors() -> dict[str, str]:
    return style.CELL_TYPE_COLORS[DATASET]


def carriers(k: int, side: str) -> tuple[str, ...]:
    return CFG.carrier_labels(k, side)


def programs() -> list[int]:
    return list(CFG.carriers)


def side_rows() -> list[tuple[int, str]]:
    return [(k, s) for k in programs() for s in H.SIDES]


def oracle(v: np.ndarray) -> np.ndarray:
    """Each cell's (sample x fine type) mean: what a fine-label pool holds."""
    c = cells()
    return (
        pd.Series(v).groupby([c["sample"], c["celltype"]]).transform("mean").to_numpy()
    )


def span(v: np.ndarray, mode: str = "q99.9") -> float:
    """Half-width of a symmetric colour scale: the 99.9th percentile of |v|,
    or 0.3 of the max (the legacy PaCMAP scale)."""
    if mode == "legacy":
        return 0.3 * float(np.abs(v).max())
    return float(np.percentile(np.abs(v), 99.9)) or 1.0


def scatter_values(ax, v, lim=None, title=None, cmap=None):
    """Cells on the PaCMAP coloured by `v`, largest |v| drawn on top."""
    xy = pacmap()
    order = np.argsort(np.abs(v))
    lim = lim or span(v)
    points = ax.scatter(
        xy[order, 0],
        xy[order, 1],
        c=v[order],
        s=POINT,
        cmap=cmap or style.CMAPS["diverging"],
        norm=Normalize(-lim, lim),
        linewidths=0,
        rasterized=True,
    )
    bare(ax, title)
    return points


def scatter_labels(ax, labels, colors: dict, title=None, background="0.88"):
    """Cells on the PaCMAP coloured by label; labels not in `colors` grey,
    drawn beneath."""
    xy = pacmap()
    rest = ~np.isin(labels, list(colors))
    ax.scatter(*xy[rest].T, c=background, s=POINT, linewidths=0, rasterized=True)
    for label, color in colors.items():
        m = labels == label
        size = POINT if m.sum() > 20000 else 4 * POINT if m.sum() > 2000 else 1.5
        ax.scatter(
            *xy[m].T,
            c=color,
            s=size,
            linewidths=0,
            rasterized=True,
            label=f"{label} ({m.sum():,})",
        )
    bare(ax, title)


def bare(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=style.FONT_SIZE_DENSE)


def heatmap(
    ax,
    frame: pd.DataFrame,
    cmap,
    vmin=None,
    vmax=None,
    label="",
    xticks=True,
    yfont=None,
    cbar=True,
):
    im = ax.imshow(
        frame.to_numpy(float),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(
        range(len(frame.index)), frame.index, fontsize=yfont or style.FONT_SIZE_DENSE
    )
    if xticks:
        ax.set_xticks(
            range(len(frame.columns)),
            frame.columns,
            fontsize=style.FONT_SIZE_DENSE,
            rotation=90,
        )
    else:
        ax.set_xticks([])
    if not cbar:
        return im
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, fontsize=style.FONT_SIZE_DENSE)
    cb.ax.tick_params(labelsize=style.FONT_SIZE_DENSE)
    return im


def lineage_separators(ax, labels: list[str], horizontal=True):
    c = cells()
    parent = pd.Series(c["lineage"], index=c["celltype"]).groupby(level=0).first()
    prev = None
    for i, t in enumerate(labels):
        if prev is not None and parent[t] != prev:
            (ax.axhline if horizontal else ax.axvline)(
                i - 0.5, color="k", lw=style.LINE_WIDTH_THIN
            )
        prev = parent[t]


def program_label(k: int) -> str:
    return CFG.program_names[k]


def figure(width_mm: float, height_mm: float, **kwargs):
    return plt.figure(
        figsize=(width_mm * style.MM, height_mm * style.MM),
        layout="constrained",
        **kwargs,
    )
