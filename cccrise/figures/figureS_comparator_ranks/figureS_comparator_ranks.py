"""Supplementary figure: how each comparator selected its rank, per dataset
and annotation (rows), from the selection traces its own-rank fit keeps
(`benchmark.fit_comparators`; `benchmark_fits_<dataset><suffix>.nc`).

- Pseudobulk CPD: core consistency (CORCONDIA) by rank, with the 85% line;
  open markers are ranks with a collinear component pair; the rank is the
  last one before either rule fails. Values below the axis floor are drawn
  on it.
- Tensor-cell2cell: mean normalised reconstruction error by rank; the rank
  is the elbow (kneedle).
- STACCato: the leading eigenvalue shares of each mode's unfolding (log),
  with the 1% line; each mode's rank is the number above it.

A row whose fits have not been run is left blank."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ...benchmark import CORCONDIA_THRESHOLD, STACCATO_VAREXP, read_fits
from ...runners.datasets import DATASETS, VALIDATION_DIR
from .. import common, style

ROWS = (
    ("alad", "", "ALAD coarse"),
    ("alad", "_fine", "ALAD fine"),
    ("ibd", "", "IBD lineage"),
    ("ibd", "_fine", "IBD fine"),
)
METHODS = ("Pseudobulk CPD", "Tensor-cell2cell", "STACCato")
STAC_MODES = ("sample", "LR", "sender", "receiver")
CC_FLOOR = -20.0
COLOR = style.COLORS["emphasis"]


def _fits(dataset, suffix):
    path = VALIDATION_DIR / DATASETS[dataset].benchmark_fits_filename(suffix)
    if not Path(path).exists():
        return None, None
    fits, info, _ = read_fits(Path(path))
    return {f.method: f for f in fits if f.rank_mode == "own"}, info


def _selected(ax, rank):
    ax.axvline(rank, color=style.COLORS["reference"], ls=":", lw=style.LINE_WIDTH_THIN)


def _pb(ax, fit):
    cc = np.asarray(fit.details["core_consistency"], float)
    collinear = np.asarray(fit.details["max_modes_collinear"]) > 0
    ranks = np.arange(1, len(cc) + 1)
    shown = np.maximum(cc, CC_FLOOR)
    ax.plot(ranks, shown, color=COLOR, lw=style.LINE_WIDTH, zorder=1)
    ax.scatter(
        ranks, shown, s=style.POINT_SIZE, zorder=2, lw=0.8, edgecolor=COLOR,
        facecolor=[("white" if c else COLOR) for c in collinear],
    )  # fmt: skip
    ax.axhline(CORCONDIA_THRESHOLD, color=style.COLORS["up"], lw=style.LINE_WIDTH_THIN)
    _selected(ax, int(fit.rank))
    ax.set_ylim(CC_FLOOR - 5, 105)
    ax.set_ylabel("core consistency (%)")


def _tc2c(ax, fit):
    errors = np.asarray(fit.details["elbow_errors"], float)
    ranks = np.arange(1, len(errors) + 1)
    ax.plot(ranks, errors, color=COLOR, lw=style.LINE_WIDTH, marker="o", ms=2)
    _selected(ax, int(fit.rank))
    ax.set_ylabel("normalised error")


def _stac(ax, fit):
    palette = style.CATEGORICAL
    for mode, props, rank, color in zip(
        STAC_MODES, fit.details["top_proportions"], fit.details["core_shape"],
        palette[: len(STAC_MODES)], strict=True,
    ):  # fmt: skip
        props = np.asarray(props, float)
        ax.plot(
            np.arange(1, len(props) + 1), props, color=color, lw=style.LINE_WIDTH,
            label=f"{mode} ({rank})",
        )  # fmt: skip
    ax.axhline(
        STACCATO_VAREXP / 100, color=style.COLORS["up"], lw=style.LINE_WIDTH_THIN
    )
    ax.set_yscale("log")
    ax.set_ylabel("eigenvalue share")
    ax.legend(fontsize=style.FONT_SIZE_DENSE - 1, loc="upper right", ncol=2)


DRAW = {"Pseudobulk CPD": _pb, "Tensor-cell2cell": _tc2c, "STACCato": _stac}


def _row(fig, dataset, suffix, label):
    fits, info = _fits(dataset, suffix)
    axes = fig.subplots(1, len(METHODS))
    types = f", {info['tensor_shape'][1]} types" if info else ""
    fig.suptitle(f"{label}{types}", fontsize=style.FONT_SIZE_TITLE, fontweight="normal")
    for ax, method in zip(axes, METHODS, strict=True):
        ax.grid(False)
        if fits is None:
            ax.set_axis_off()
            continue
        DRAW[method](ax, fits[method])
        ax.set_title(
            f"{method}: {fits[method].rank}", fontsize=style.FONT_SIZE_DENSE + 1
        )
        ax.set_xlabel("rank" if method != "STACCato" else "component")


def _panel(dataset, suffix, label):
    # one level down, so the row's title sits under the panel letter
    return SimpleNamespace(
        plot=lambda fig: _row(fig.subfigures(1, 1), dataset, suffix, label)
    )


PANELS = {f"{d}{s}": _panel(d, s, label) for d, s, label in ROWS}
LAYOUT = [[name] for name in PANELS]


def makeFigure():
    return common.compose(
        (style.FIGURE_WIDTH_DOUBLE, style.FIGURE_HEIGHT_MAX), LAYOUT, PANELS
    )
